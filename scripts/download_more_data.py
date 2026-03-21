#!/usr/bin/env python3
"""
Скрипт для докачки данных для обучения Show-o.

Использование:
    python scripts/download_more_data.py --dataset cc12m --missing-only
    python scripts/download_more_data.py --dataset cc12m --start 109 --end 200
    python scripts/download_more_data.py --dataset laion-aesthetic --count 100

Источники данных:
- CC12M: https://huggingface.co/datasets/pixparse/cc12m-wds (WebDataset формат)
- LAION-Aesthetics: https://huggingface.co/datasets/laion/laion2B-en-aesthetic
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests
from tqdm import tqdm

# Базовые пути
DATA_DIR = Path("/home/jovyan/vasiliev/notebooks/Show-o/data")
CC12M_DIR = DATA_DIR / "cc12m"
LAION_DIR = DATA_DIR / "laion_aesthetic"

# URL шаблоны
CC12M_URL_TEMPLATE = "https://huggingface.co/datasets/pixparse/cc12m-wds/resolve/main/cc12m-train-{shard:04d}.tar"
# Альтернативный источник CC12M через img2dataset
CC12M_IMG2DATASET_URL = "https://huggingface.co/datasets/ChristophSchuhmann/improved_aesthetics_6.5plus"


def get_existing_cc12m_shards():
    """Получает список существующих шардов CC12M."""
    if not CC12M_DIR.exists():
        return set()
    
    shards = set()
    for f in CC12M_DIR.glob("cc12m-train-*.tar"):
        try:
            shard_num = int(f.stem.split("-")[-1])
            shards.add(shard_num)
        except ValueError:
            continue
    return shards


def find_missing_cc12m_shards(max_shard=108):
    """Находит отсутствующие шарды в диапазоне 0-max_shard."""
    existing = get_existing_cc12m_shards()
    expected = set(range(max_shard + 1))
    missing = expected - existing
    return sorted(missing)


def download_cc12m_shard(shard_num: int, force: bool = False) -> bool:
    """Скачивает один шард CC12M."""
    output_path = CC12M_DIR / f"cc12m-train-{shard_num:04d}.tar"
    
    if output_path.exists() and not force:
        print(f"[SKIP] Shard {shard_num:04d} already exists")
        return True
    
    url = CC12M_URL_TEMPLATE.format(shard=shard_num)
    
    try:
        print(f"[DOWNLOAD] Shard {shard_num:04d} from {url}")
        
        # Используем wget для надёжности
        cmd = [
            "wget", "-q", "--show-progress",
            "-O", str(output_path),
            url
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            print(f"[ERROR] Failed to download shard {shard_num:04d}: {result.stderr}")
            # Удаляем битый файл
            if output_path.exists():
                output_path.unlink()
            return False
        
        print(f"[OK] Shard {shard_num:04d} downloaded")
        return True
        
    except Exception as e:
        print(f"[ERROR] Exception downloading shard {shard_num:04d}: {e}")
        return False


def download_cc12m_shards(shards: list, max_workers: int = 4):
    """Скачивает несколько шардов параллельно."""
    CC12M_DIR.mkdir(parents=True, exist_ok=True)
    
    print(f"Downloading {len(shards)} CC12M shards...")
    
    successful = 0
    failed = []
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(download_cc12m_shard, shard): shard for shard in shards}
        
        for future in tqdm(as_completed(futures), total=len(futures), desc="Downloading"):
            shard = futures[future]
            try:
                if future.result():
                    successful += 1
                else:
                    failed.append(shard)
            except Exception as e:
                print(f"[ERROR] Shard {shard} exception: {e}")
                failed.append(shard)
    
    print(f"\nCompleted: {successful}/{len(shards)} shards")
    if failed:
        print(f"Failed shards: {failed}")
    
    return successful, failed


def download_laion_aesthetic_via_img2dataset(num_samples: int = 100000, output_dir: Path = None):
    """
    Скачивает LAION-Aesthetics через img2dataset.
    Требует установки: pip install img2dataset
    """
    if output_dir is None:
        output_dir = LAION_DIR
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Downloading LAION-Aesthetics ({num_samples} samples) to {output_dir}")
    print("This requires img2dataset to be installed: pip install img2dataset")
    
    # Создаём parquet файл с URL изображений
    # Для LAION используем HuggingFace datasets
    try:
        from datasets import load_dataset
        
        print("Loading LAION-Aesthetics metadata from HuggingFace...")
        ds = load_dataset(
            "laion/laion2B-en-aesthetic",
            split=f"train[:{num_samples}]",
            streaming=False
        )
        
        # Сохраняем как parquet для img2dataset
        parquet_path = output_dir / "urls.parquet"
        ds.to_parquet(str(parquet_path))
        
        print(f"Saved {len(ds)} URLs to {parquet_path}")
        print("\nRun img2dataset manually:")
        print(f"  img2dataset --url_list {parquet_path} --output_folder {output_dir}/images \\")
        print("    --input_format parquet --url_col URL --caption_col TEXT \\")
        print("    --output_format webdataset --processes_count 16 --thread_count 64 \\")
        print("    --image_size 512 --resize_mode center_crop")
        
        return True
        
    except ImportError:
        print("Please install datasets: pip install datasets")
        return False
    except Exception as e:
        print(f"Error: {e}")
        return False


def create_webdataset_from_folder(image_folder: Path, output_path: Path, images_per_shard: int = 10000):
    """Создаёт WebDataset .tar файлы из папки с изображениями."""
    try:
        import webdataset as wds
        from PIL import Image
        import io
        import json
        
        image_files = list(image_folder.glob("**/*.jpg")) + \
                     list(image_folder.glob("**/*.png")) + \
                     list(image_folder.glob("**/*.jpeg"))
        
        print(f"Found {len(image_files)} images in {image_folder}")
        
        output_path.mkdir(parents=True, exist_ok=True)
        
        shard_idx = 0
        sample_idx = 0
        current_shard = None
        
        for img_path in tqdm(image_files, desc="Creating shards"):
            if sample_idx % images_per_shard == 0:
                if current_shard:
                    current_shard.close()
                shard_path = output_path / f"shard-{shard_idx:05d}.tar"
                current_shard = wds.TarWriter(str(shard_path))
                shard_idx += 1
            
            try:
                with open(img_path, "rb") as f:
                    img_bytes = f.read()
                
                # Пытаемся найти caption файл
                caption_path = img_path.with_suffix(".txt")
                if caption_path.exists():
                    caption = caption_path.read_text().strip()
                else:
                    caption = ""
                
                sample = {
                    "__key__": f"{sample_idx:08d}",
                    "jpg": img_bytes,
                    "txt": caption,
                }
                current_shard.write(sample)
                sample_idx += 1
                
            except Exception as e:
                print(f"Error processing {img_path}: {e}")
                continue
        
        if current_shard:
            current_shard.close()
        
        print(f"Created {shard_idx} shards with {sample_idx} total samples")
        return True
        
    except ImportError:
        print("Please install webdataset: pip install webdataset")
        return False


def main():
    parser = argparse.ArgumentParser(description="Download additional training data for Show-o")
    parser.add_argument("--dataset", choices=["cc12m", "laion-aesthetic", "both"], default="cc12m",
                       help="Which dataset to download")
    parser.add_argument("--missing-only", action="store_true",
                       help="Download only missing CC12M shards (0-108)")
    parser.add_argument("--start", type=int, default=0,
                       help="Start shard number for CC12M")
    parser.add_argument("--end", type=int, default=108,
                       help="End shard number for CC12M")
    parser.add_argument("--count", type=int, default=100000,
                       help="Number of samples for LAION-Aesthetic")
    parser.add_argument("--workers", type=int, default=4,
                       help="Number of parallel download workers")
    parser.add_argument("--list-missing", action="store_true",
                       help="Only list missing shards, don't download")
    
    args = parser.parse_args()
    
    if args.dataset in ["cc12m", "both"]:
        if args.list_missing:
            missing = find_missing_cc12m_shards(args.end)
            print(f"Missing CC12M shards: {missing}")
            print(f"Total missing: {len(missing)}")
            return
        
        if args.missing_only:
            shards = find_missing_cc12m_shards(args.end)
            if not shards:
                print("No missing shards found!")
                return
            print(f"Found {len(shards)} missing shards: {shards}")
        else:
            shards = list(range(args.start, args.end + 1))
        
        download_cc12m_shards(shards, max_workers=args.workers)
    
    if args.dataset in ["laion-aesthetic", "both"]:
        download_laion_aesthetic_via_img2dataset(num_samples=args.count)


if __name__ == "__main__":
    main()







