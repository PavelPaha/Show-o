#!/usr/bin/env python3
"""
Проверка обрезания последовательностей в T2I обучении
Проверяет не обрезаются ли маскированные токены изображения
"""

import sys
sys.path.insert(0, "/home/jovyan/vasiliev/notebooks/Show-o")

import torch
import os
os.environ["CUDA_HOME"] = "/home/jovyan/vasiliev/notebooks/Show-o/cuda_fake"

from omegaconf import OmegaConf
from training.data import Text2ImageDataset
from training.prompting_utils import UniversalPrompting
from transformers import AutoTokenizer
from models import MAGVITv2, get_mask_chedule
from training.utils import mask_or_random_replace_tokens

print("=" * 80)
print("🔍 ПРОВЕРКА ОБРЕЗАНИЯ ПОСЛЕДОВАТЕЛЬНОСТЕЙ В T2I")
print("=" * 80)
print()

# Загружаем конфиг
config = OmegaConf.load("configs/showo_mmu_moe.yaml")
max_seq_length = config.dataset.preprocessing.max_seq_length
resolution = config.dataset.preprocessing.resolution

print(f"📋 Параметры:")
print(f"   max_seq_length: {max_seq_length}")
print(f"   resolution: {resolution}")
print()

# Инициализация компонентов
tokenizer = AutoTokenizer.from_pretrained(config.model.showo.llm_model_path)
uni_prompting = UniversalPrompting(
    tokenizer,
    max_text_len=max_seq_length,
    special_tokens=("<|soi|>", "<|eoi|>", "<|sov|>", "<|eov|>", "<|t2i|>", "<|mmu|>", "<|t2v|>"),
)

vq_model = MAGVITv2.from_pretrained(config.model.vq_model.vq_model_name)
vq_model.eval()

# mask_schedule - используем "cosine" как в референсе
mask_schedule = get_mask_chedule("cosine")
# mask_id - используем специальный токен для маскирования
# Проверяем какие токены доступны
if "<|mask|>" in uni_prompting.sptids_dict:
    mask_id = uni_prompting.sptids_dict["<|mask|>"]
else:
    # Используем первый доступный специальный токен или создаём фиктивный
    mask_id = len(uni_prompting.text_tokenizer) + 8192  # После image tokens

# Создаём dataset
dataset = Text2ImageDataset(
    train_shards_path_or_url=config.dataset.params.train_t2i_shards_path_or_url,
    tokenizer=None,  # raw texts
    max_seq_length=max_seq_length,
    num_train_examples=1000,  # Проверяем 1000 сэмплов
    per_gpu_batch_size=1,
    global_batch_size=1,
    num_workers=1,
    resolution=resolution,
    shuffle_buffer_size=100,
    pin_memory=False,
    persistent_workers=False,
    external_caption_path=config.dataset.params.external_caption_path,
    external_cc12m_caption_path=config.dataset.params.external_cc12m_caption_path,
)

dataloader = dataset.train_dataloader

print("📥 Обрабатываю 1000 сэмплов...\n")

# Статистика
total_samples = 0
truncated_samples = 0
image_tokens_truncated = 0
text_truncated = 0
max_text_len = 0
max_total_len = 0
min_total_len = float('inf')

# Детальная статистика
text_lengths = []
image_token_lengths = []
total_lengths = []
truncation_details = []

for i, batch in enumerate(dataloader):
    if i >= 1000:
        break
    
    total_samples += 1
    
    pixel_values = batch["images"]
    texts = batch["input_ids"]
    
    # Подготовка последовательности как в train.py
    with torch.no_grad():
        # Кодируем изображение в токены
        image_tokens_ori = vq_model.get_code(pixel_values)
        image_tokens_ori = image_tokens_ori + len(uni_prompting.text_tokenizer)
        
        # Маскируем токены
        input_ids_img, labels_img, loss_weight, mask_prob = mask_or_random_replace_tokens(
            image_tokens_ori,
            mask_id,
            config,
            mask_schedule=mask_schedule,
            is_train=True,
        )
        
        # Формируем последовательность
        input_ids, masks, labels = uni_prompting((texts, input_ids_img, labels_img), "t2i")
    
    # Анализ последовательности
    seq_len = input_ids.shape[1]
    total_lengths.append(seq_len)
    max_total_len = max(max_total_len, seq_len)
    min_total_len = min(min_total_len, seq_len)
    
    # Проверяем длину текста
    text_tokens = uni_prompting.text_tokenizer(texts[0])['input_ids']
    text_len = len(text_tokens[0])
    text_lengths.append(text_len)
    max_text_len = max(max_text_len, text_len)
    
    # Проверяем длину image tokens
    image_len = image_tokens_ori.shape[1]
    image_token_lengths.append(image_len)
    
    # Проверяем обрезание
    # Последовательность: [<|t2i|>] [text] [<|soi|>] [image tokens] [<|eoi|>]
    # text занимает до max_seq_length
    # image tokens начинаются с max_seq_length + 1
    
    soi_pos = None
    eoi_pos = None
    for j, token_id in enumerate(input_ids[0]):
        if token_id == uni_prompting.sptids_dict['<|soi|>']:
            soi_pos = j
        if token_id == uni_prompting.sptids_dict['<|eoi|>']:
            eoi_pos = j
            break
    
    if soi_pos is None or eoi_pos is None:
        print(f"⚠️  Sample {i}: Не найдены <|soi|> или <|eoi|> токены!")
        continue
    
    # Проверяем обрезание текста
    text_expected_end = soi_pos
    if text_len > max_seq_length:
        text_truncated += 1
        truncation_details.append({
            'sample': i,
            'type': 'text',
            'original_len': text_len,
            'max_allowed': max_seq_length
        })
    
    # Проверяем обрезание image tokens
    image_tokens_start = soi_pos + 1
    image_tokens_end = eoi_pos
    image_tokens_in_seq = image_tokens_end - image_tokens_start
    image_tokens_expected = image_tokens_ori.shape[1]
    
    if image_tokens_in_seq < image_tokens_expected:
        image_tokens_truncated += 1
        truncated_samples += 1
        truncation_details.append({
            'sample': i,
            'type': 'image',
            'original_len': image_tokens_expected,
            'actual_len': image_tokens_in_seq,
            'lost_tokens': image_tokens_expected - image_tokens_in_seq
        })
    
    # Проверяем общую длину последовательности
    if seq_len > max_seq_length + image_tokens_expected + 5:  # +5 для special tokens
        truncated_samples += 1
    
    if (i + 1) % 100 == 0:
        print(f"  Обработано: {i+1}/1000")

print()
print("=" * 80)
print("📊 РЕЗУЛЬТАТЫ")
print("=" * 80)
print()

print(f"✅ Всего обработано сэмплов: {total_samples}")
print()

print(f"📏 Длины последовательностей:")
print(f"   Минимальная: {min_total_len}")
print(f"   Максимальная: {max_total_len}")
print(f"   Средняя: {sum(total_lengths) / len(total_lengths):.1f}")
print()

print(f"📝 Длины текстов:")
print(f"   Максимальная: {max_text_len}")
print(f"   Средняя: {sum(text_lengths) / len(text_lengths):.1f}")
print(f"   max_seq_length в конфиге: {max_seq_length}")
if max_text_len > max_seq_length:
    print(f"   ⚠️  Тексты обрезаются! (max={max_text_len} > {max_seq_length})")
else:
    print(f"   ✅ Тексты не обрезаются")
print()

print(f"🖼️  Длины image tokens:")
print(f"   Минимальная: {min(image_token_lengths)}")
print(f"   Максимальная: {max(image_token_lengths)}")
print(f"   Средняя: {sum(image_token_lengths) / len(image_token_lengths):.1f}")
print(f"   Ожидаемая для {resolution}x{resolution}: ~256 токенов")
print()

print(f"✂️  ОБРЕЗАНИЕ:")
print(f"   Тексты обрезаны: {text_truncated}/{total_samples} ({100*text_truncated/total_samples:.1f}%)")
print(f"   Image tokens обрезаны: {image_tokens_truncated}/{total_samples} ({100*image_tokens_truncated/total_samples:.1f}%)")
print(f"   Всего проблемных сэмплов: {truncated_samples}/{total_samples} ({100*truncated_samples/total_samples:.1f}%)")
print()

if image_tokens_truncated > 0:
    print("❌ КРИТИЧЕСКАЯ ПРОБЛЕМА: Image tokens обрезаются!")
    print("   Это означает что маскированные токены теряются!")
    print("   Модель не может учиться на обрезанных токенах!")
    print()
    print("   Первые 10 случаев обрезания image tokens:")
    image_truncs = [d for d in truncation_details if d['type'] == 'image'][:10]
    for detail in image_truncs:
        print(f"      Sample {detail['sample']}: потеряно {detail['lost_tokens']} токенов "
              f"({detail['original_len']} → {detail['actual_len']})")
    print()
    print("💡 РЕШЕНИЕ:")
    print("   1. Увеличь max_seq_length в конфиге")
    print("   2. Или уменьши resolution (меньше image tokens)")
    print("   3. Или используй более агрессивное обрезание текста")
else:
    print("✅ Image tokens НЕ обрезаются - всё в порядке!")

if text_truncated > 0:
    print()
    print("⚠️  Тексты обрезаются, но это нормально если не критично")
    print(f"   {text_truncated} сэмплов имеют текст длиннее {max_seq_length}")

print()
print("=" * 80)

