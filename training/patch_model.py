# coding=utf-8
from typing import
from moe import MoE
from moe_mlflow_logger import MoEMLflowLogger
from moe_visualization import MoEVisualizer


def patch_model_with_moe(
    model,
    moe_config,
    mlflow_client,
    mlflow_run_id,
    special_tokens=None,
):
    count_layers_to_patch = moe_config["count_layers_to_patch"]
    num_experts = int(moe_config["num_experts"])
    top_k = int(moe_config["top_k"])
    total_layers = len(model.showo.model.layers)
    print(f"🔧 Патчим модель с MoE:")
    print(f"   Всего слоев в модели: {total_layers}")
    print(f"   Количество экспертов: {num_experts}")
    print(f"   Top-K: {top_k}")
    print(f"   Слоев для патчинга: {count_layers_to_patch}")
    config_phi = model.showo.config
    patched_layers = []
    num_experts = moe_config["num_experts"]   
    moe_exists = False 
    
    for layer_idx, layer in list(enumerate(model.showo.model.layers))[::-1]:
        if count_layers_to_patch == 0:
            break
        
        if hasattr(layer, 'mlp'):
            print(f"  → Слой {layer_idx}")
            patched_layers.append(layer_idx)
            original_mlp = layer.mlp
            mlflow_logger = MoEMLflowLogger(mlflow_client=mlflow_client, mlflow_run_id=mlflow_run_id)
            visualizer = MoEVisualizer(num_experts=num_experts, layer_id=layer_idx)
            moe_layer = MoE(
                config=config_phi,
                moe_config=moe_config,
                template_mlp=original_mlp,
                mlflow_logger=mlflow_logger,
                visualizer=visualizer,
                layer_idx=layer_idx,
                special_tokens=special_tokens
            )
            moe_layer.to(next(original_mlp.parameters()).device)
            layer.mlp = moe_layer
            moe_exists = True
            count_layers_to_patch -= 1
            layer.requires_grad = True
        else:
            layer.requires_grad = False

            
    if not moe_exists:
        raise Exception("No moe layers created")
    print(f"✓ Заменено слоев: {len(patched_layers)} из {total_layers} ({100*len(patched_layers)/total_layers:.1f}%)")
    print(f"✓ Слои с MoE: {patched_layers}")
    print(f"✓ Слои с оригинальным FFN (предобученным): {[i for i in range(total_layers) if i not in patched_layers]}")
    
    return model