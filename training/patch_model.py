# coding=utf-8
from typing import Optional
import re
from moe import MoE
from moe_comet_logger import MoECometLogger
from moe_visualization import MoEVisualizer


def patch_model_with_moe(
    model,
    moe_config,
    comet_experiment=None,
    special_tokens=None,
):
    count_layers_to_patch = moe_config["count_layers_to_patch"]
    num_experts = int(moe_config["num_experts"])
    top_k = int(moe_config["top_k"])
    total_layers = len(model.showo.model.layers)
    freeze_base = moe_config.get("freeze_base_model", True)  # По умолчанию замораживаем базу
    
    print(f"🔧 Патчим модель с MoE:")
    print(f"   Всего слоев в модели: {total_layers}")
    print(f"   Количество экспертов: {num_experts}")
    print(f"   Top-K: {top_k}")
    print(f"   Слоев для патчинга: {count_layers_to_patch}")
    print(f"   Заморозить базовую модель: {freeze_base}")
    
    config_phi = model.showo.config
    patched_layers = []
    num_experts = moe_config["num_experts"]   
    moe_exists = False 
    
    # ШАГ 1: Замораживаем или размораживаем базовую модель
    if freeze_base:
        frozen_count = 0
        for name, param in model.named_parameters():
            param.requires_grad = False
            frozen_count += 1
        print(f"   ❄️  Заморожено {frozen_count} параметров базовой модели")
    else:
        # Явно размораживаем базовые параметры (кроме MoE, которые будут обработаны позже)
        unfrozen_count = 0
        for name, param in model.named_parameters():
            # Пропускаем MoE параметры - они будут обработаны в ШАГ 3
            if not any(x in name for x in ["mlp.experts", "mlp.gate", "mlp.alpha"]):
                param.requires_grad = True
                unfrozen_count += 1
        print(f"   🔥 Разморожено {unfrozen_count} параметров базовой модели")
    
    # ШАГ 2: Патчим MLP слои на MoE
    layers_to_patch = count_layers_to_patch
    for layer_idx, layer in list(enumerate(model.showo.model.layers))[::-1]:
        if layers_to_patch == 0:
            break
        
        if hasattr(layer, 'mlp'):
            print(f"  → Слой {layer_idx}")
            patched_layers.append(layer_idx)
            original_mlp = layer.mlp
            comet_logger = MoECometLogger(comet_experiment=comet_experiment)
            visualizer = MoEVisualizer(num_experts=num_experts, layer_id=layer_idx)
            moe_layer = MoE(
                config=config_phi,
                moe_config=moe_config,
                template_mlp=original_mlp,
                comet_logger=comet_logger,
                visualizer=visualizer,
                layer_idx=layer_idx,
                special_tokens=special_tokens
            )
            moe_layer.to(next(original_mlp.parameters()).device)
            layer.mlp = moe_layer
            moe_exists = True
            layers_to_patch -= 1
    
    # ШАГ 3: Размораживаем ТОЛЬКО MoE параметры
    moe_unfrozen_count = 0
    for name, param in model.named_parameters():
        # Размораживаем только MoE компоненты: experts, gate, alpha
        if any(x in name for x in ["mlp.experts", "mlp.gate", "mlp.alpha"]):
            param.requires_grad = True
            moe_unfrozen_count += 1
    
    print(f"   🔥 Разморожено {moe_unfrozen_count} параметров MoE")
            
    if not moe_exists:
        raise Exception("No moe layers created")
    
    # Подсчитываем статистику
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    
    # Проверка: считаем параметры по категориям
    base_trainable = sum(p.numel() for name, p in model.named_parameters() 
                        if p.requires_grad and not any(x in name for x in ["mlp.experts", "mlp.gate", "mlp.alpha"]))
    moe_trainable = sum(p.numel() for name, p in model.named_parameters() 
                       if p.requires_grad and any(x in name for x in ["mlp.experts", "mlp.gate", "mlp.alpha"]))
    
    print(f"✓ Заменено слоев: {len(patched_layers)} из {total_layers} ({100*len(patched_layers)/total_layers:.1f}%)")
    print(f"✓ Слои с MoE: {patched_layers}")
    print(f"✓ Слои с оригинальным FFN: {[i for i in range(total_layers) if i not in patched_layers]}")
    print(f"✓ Обучаемых параметров: {trainable_params:,} / {total_params:,} ({100*trainable_params/total_params:.2f}%)")
    print(f"   - Базовые параметры: {base_trainable:,} {'✅ разморожены' if base_trainable > 0 and not freeze_base else '❄️ заморожены'}")
    print(f"   - MoE параметры: {moe_trainable:,} ✅ разморожены")
    
    # Валидация: если freeze_base=False, должны быть разморожены базовые параметры
    if not freeze_base and base_trainable == 0:
        print(f"⚠️  ВНИМАНИЕ: freeze_base_model=False, но базовые параметры не разморожены!")
        print(f"   Проверьте, что модель загружена с правильными настройками.")
    
    return model