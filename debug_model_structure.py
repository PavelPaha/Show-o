#!/usr/bin/env python3
"""
Скрипт для отладки структуры модели Show-o
"""

import torch
from omegaconf import OmegaConf
from models.modeling.modeling_showo import Showo


def get_config():
    """Загружает конфигурацию"""
    cli_conf = OmegaConf.from_cli()
    yaml_conf = OmegaConf.load(cli_conf.config)
    conf = OmegaConf.merge(yaml_conf, cli_conf)
    return conf


def debug_model_structure(model):
    """Отлаживает структуру модели"""
    print("=== Структура модели Show-o ===")
    
    # Проверяем основную структуру
    print(f"Тип модели: {type(model)}")
    print(f"Атрибуты модели: {[attr for attr in dir(model) if not attr.startswith('_')]}")
    
    if hasattr(model, 'showo'):
        print(f"\n=== Структура showo ===")
        print(f"Тип showo: {type(model.showo)}")
        print(f"Атрибуты showo: {[attr for attr in dir(model.showo) if not attr.startswith('_')]}")
        
        if hasattr(model.showo, 'model'):
            print(f"\n=== Структура showo.model ===")
            print(f"Тип model: {type(model.showo.model)}")
            print(f"Атрибуты model: {[attr for attr in dir(model.showo.model) if not attr.startswith('_')]}")
            
            if hasattr(model.showo.model, 'layers'):
                print(f"\n=== Структура слоев ===")
                print(f"Количество слоев: {len(model.showo.model.layers)}")
                
                for i, layer in enumerate(model.showo.model.layers):
                    print(f"\nСлой {i}:")
                    print(f"  Тип: {type(layer)}")
                    print(f"  Атрибуты: {[attr for attr in dir(layer) if not attr.startswith('_')]}")
                    
                    # Проверяем есть ли ffn или mlp
                    if hasattr(layer, 'ffn'):
                        print(f"  ✅ Есть ffn: {type(layer.ffn)}")
                    elif hasattr(layer, 'mlp'):
                        print(f"  ✅ Есть mlp: {type(layer.mlp)}")
                    else:
                        print(f"  ❌ Нет ffn или mlp")
                        
                    # Показываем все атрибуты слоя
                    for attr in dir(layer):
                        if not attr.startswith('_') and not callable(getattr(layer, attr)):
                            value = getattr(layer, attr)
                            if hasattr(value, '__class__'):
                                print(f"    {attr}: {type(value)}")


def main():
    """Основная функция"""
    print("=== Отладка структуры модели Show-o ===")
    
    # Загружаем конфигурацию
    config = get_config()
    print(f"Конфигурация загружена из {config.config}")
    
    # Загружаем модель
    print("Загружаем модель...")
    model = Showo.from_pretrained(config.model.showo.pretrained_model_path)
    print("✅ Модель загружена")
    
    # Отлаживаем структуру
    debug_model_structure(model)


if __name__ == "__main__":
    main()

