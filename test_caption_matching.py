#!/usr/bin/env python3
"""
Тест: проверяем что caption'ы правильно сопоставляются с изображениями в T2I задаче
"""

import sys
sys.path.insert(0, "/home/jovyan/vasiliev/notebooks/Show-o")

from training.data import Text2ImageDataset
from omegaconf import OmegaConf

print("🔍 Тест сопоставления caption'ов и изображений для T2I\n")

# Загружаем конфиг
config = OmegaConf.load("configs/showo_mmu_moe.yaml")

# Создаём dataset как в реальном коде
dataset = Text2ImageDataset(
    train_shards_path_or_url=config.dataset.params.train_t2i_shards_path_or_url,
    tokenizer=None,  # we want to get raw texts
    max_seq_length=config.dataset.preprocessing.max_seq_length,
    num_train_examples=100,  # Маленький тест
    per_gpu_batch_size=2,
    global_batch_size=2,
    num_workers=1,  # Минимум 1 для WebDataset
    resolution=config.dataset.preprocessing.resolution,
    shuffle_buffer_size=1,  # Минимальный shuffle для теста
    pin_memory=False,
    persistent_workers=False,
    external_caption_path=config.dataset.params.external_caption_path,
    external_cc12m_caption_path=config.dataset.params.external_cc12m_caption_path,
)

print("✅ Dataset создан\n")
print("📥 Загружаю первые 5 батчей...\n")

dataloader = dataset.train_dataloader
matched_count = 0
empty_caption_count = 0

for i, batch in enumerate(dataloader):
    if i >= 5:
        break
    
    images = batch["images"]
    texts = batch["input_ids"]  # Это список строк когда tokenizer=None
    
    print(f"Батч {i+1}:")
    print(f"  Images shape: {images.shape}")
    print(f"  Texts count: {len(texts)}")
    
    for j, text in enumerate(texts):
        if isinstance(text, str):
            if len(text.strip()) > 0:
                matched_count += 1
                print(f"  ✅ Sample {j}: caption length={len(text)}, preview: {text[:60]}...")
            else:
                empty_caption_count += 1
                print(f"  ❌ Sample {j}: ПУСТОЙ caption!")
        else:
            print(f"  ⚠️  Sample {j}: caption не строка, тип={type(text)}")
    print()

print(f"\n📊 Итоги:")
print(f"  ✅ Сопоставлено: {matched_count}")
print(f"  ❌ Пустых caption'ов: {empty_caption_count}")

if empty_caption_count > 0:
    print("\n⚠️  ВНИМАНИЕ: Есть пустые caption'ы! Это может быть проблемой.")
    print("   Проверь что external_caption_path пустой в конфиге.")
else:
    print("\n✅ Все caption'ы заполнены!")

