#!/usr/bin/env python3
"""
Скрипт для скачивания LAION данных и создания webdataset tar shards.
Увеличивает количество данных в 5 раз (10k вместо 2k).
"""

import os
import tarfile
import io
from pathlib import Path
from datasets import load_dataset
from PIL import Image
from tqdm import tqdm
import json
import requests
from io import BytesIO

def download_laion10k(output_dir="./data/laion10k", num_samples=10000, samples_per_shard=200, resume=True):
    """
    Скачивает LAION данные и создает webdataset tar shards.
    
    Args:
        output_dir: Директория для сохранения shards
        num_samples: Общее количество образцов для скачивания
        samples_per_shard: Количество образцов в каждом shard
        resume: Продолжить с последнего шарда
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Проверяем существующие шарды для resume
    existing_shards = sorted(output_path.glob("laion10k-*.tar"))
    if resume and existing_shards:
        last_shard = existing_shards[-1].stem  # laion10k-00037
        start_shard_idx = int(last_shard.split("-")[1]) + 1
        start_sample_idx = start_shard_idx * samples_per_shard
        print(f"🔄 Продолжение с шарда {start_shard_idx} (sample {start_sample_idx})")
    else:
        start_shard_idx = 0
        start_sample_idx = 0
    
    print(f"📥 Загрузка {num_samples} образцов из LAION-aesthetics-12M...")
    
    # Загружаем датасет из Hugging Face
    dataset = load_dataset(
        "dclure/laion-aesthetics-12m-umap",
        split="train",
        streaming=True
    )
    
    num_shards = (num_samples + samples_per_shard - 1) // samples_per_shard
    
    print(f"📦 Создание {num_shards} shards по {samples_per_shard} образцов...")
    
    shard_idx = start_shard_idx
    sample_idx = start_sample_idx
    current_shard_samples = 0
    tar_path = None
    tar_file = None
    max_iterations = num_samples * 10  # Максимум 10x итераций (много URL могут быть недоступны)
    iteration = 0
    failed_downloads = 0
    skipped_for_resume = 0
    # Примерно сколько итераций нужно пропустить (учитывая failed downloads ~50%)
    skip_iterations = start_sample_idx * 2 if start_sample_idx > 0 else 0
    
    try:
        for item in tqdm(dataset, desc="Обработка образцов"):
            iteration += 1
            
            # Пропускаем итерации для resume (грубая оценка)
            if skip_iterations > 0 and skipped_for_resume < skip_iterations:
                skipped_for_resume += 1
                if skipped_for_resume % 1000 == 0:
                    print(f"  ⏭️ Пропущено {skipped_for_resume}/{skip_iterations} для resume...")
                continue
            
            # Останавливаемся если достигли нужного количества образцов
            if sample_idx >= num_samples:
                break
            
            # Останавливаемся если слишком много итераций без результата
            if iteration > max_iterations + skip_iterations:
                print(f"\n⚠️ Достигнут лимит итераций ({max_iterations}). Остановка.")
                break
            
            # Начинаем новый shard если нужно
            if current_shard_samples == 0 and tar_file is None:
                shard_name = f"laion10k-{shard_idx:05d}.tar"
                tar_path = output_path / shard_name
                tar_file = tarfile.open(tar_path, "w")
                print(f"\n📝 Создание shard {shard_idx}: {shard_name}")
            
            # Получаем изображение и текст
            try:
                # Получаем URL изображения
                url = item.get("URL", "")
                if not url:
                    continue
                
                # Получаем текст (caption)
                text = item.get("TEXT", item.get("caption", item.get("text", "")))
                if not text:
                    continue
                
                # Скачиваем изображение по URL
                try:
                    response = requests.get(url, timeout=10, stream=True)
                    response.raise_for_status()
                    image = Image.open(BytesIO(response.content))
                    
                    # Конвертируем PIL Image в bytes
                    img_buffer = io.BytesIO()
                    if image.mode != "RGB":
                        image = image.convert("RGB")
                    image.save(img_buffer, format="JPEG", quality=95)
                    img_bytes = img_buffer.getvalue()
                except Exception as img_error:
                    # Пропускаем если не удалось скачать изображение
                    failed_downloads += 1
                    if failed_downloads % 100 == 0:
                        print(f"  ⚠️ Пропущено {failed_downloads} недоступных изображений")
                    continue
                
                # Создаем имя файла
                sample_name = f"{sample_idx:08d}"
                
                # Добавляем изображение в tar
                img_info = tarfile.TarInfo(name=f"{sample_name}.jpg")
                img_info.size = len(img_bytes)
                tar_file.addfile(img_info, io.BytesIO(img_bytes))
                
                # Добавляем текст в tar
                text_bytes = text.encode("utf-8")
                txt_info = tarfile.TarInfo(name=f"{sample_name}.txt")
                txt_info.size = len(text_bytes)
                tar_file.addfile(txt_info, io.BytesIO(text_bytes))
                
                current_shard_samples += 1
                sample_idx += 1
                
                # Обновляем прогресс каждые 100 образцов
                if sample_idx % 100 == 0:
                    print(f"  ✓ Обработано {sample_idx}/{num_samples} образцов")
                
                # Закрываем shard если он заполнен
                if current_shard_samples >= samples_per_shard:
                    tar_file.close()
                    tar_file = None
                    current_shard_samples = 0
                    shard_idx += 1
                    
            except Exception as e:
                # Тихо пропускаем ошибки, не печатаем каждую
                if sample_idx % 1000 == 0:
                    print(f"\n⚠️ Ошибка при обработке (пропущено много образцов): {e}")
                continue
        
        # Закрываем последний shard
        if tar_file is not None:
            tar_file.close()
        
        print(f"\n✅ Успешно создано {shard_idx + 1} shards с {sample_idx} образцами")
        print(f"📁 Shards сохранены в: {output_path}")
        print(f"\nОбновите конфиг, добавив:")
        print(f'  - "{output_path}/laion10k-{{00000..{shard_idx:05d}}}.tar"')
        
    except Exception as e:
        print(f"\n❌ Ошибка при скачивании: {e}")
        if tar_file is not None:
            tar_file.close()
        raise


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Скачать LAION данные и создать webdataset shards")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./data/laion10k",
        help="Директория для сохранения shards (по умолчанию: ./data/laion10k)"
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=10000,
        help="Количество образцов для скачивания (по умолчанию: 10000)"
    )
    parser.add_argument(
        "--samples-per-shard",
        type=int,
        default=200,
        help="Количество образцов в каждом shard (по умолчанию: 200)"
    )
    
    args = parser.parse_args()
    
    download_laion10k(
        output_dir=args.output_dir,
        num_samples=args.num_samples,
        samples_per_shard=args.samples_per_shard
    )

