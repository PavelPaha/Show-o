import os
import tarfile
import random
from pathlib import Path
from PIL import Image
import io

def inspect_dataset(name, data_dir, output_dir, num_samples=3):
    print(f"\n{'='*20} INSPECTING {name} {'='*20}")
    data_path = Path(data_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Находим все tar файлы
    tars = list(data_path.glob("*.tar"))
    if not tars:
        print(f"❌ No .tar files found in {data_dir}")
        return

    # Берем случайный tar
    random_tar = random.choice(tars)
    print(f"📦 Reading shard: {random_tar.name}")
    
    samples = {}
    
    try:
        with tarfile.open(random_tar, "r") as tar:
            # Собираем пары файлов
            for member in tar.getmembers():
                if not member.isfile():
                    continue
                
                # Имя файла без расширения (key)
                filename = os.path.basename(member.name)
                key = os.path.splitext(filename)[0]
                ext = os.path.splitext(filename)[1].lower()
                
                if key not in samples:
                    samples[key] = {}
                
                if ext in ['.jpg', '.jpeg', '.png', '.webp']:
                    samples[key]['img'] = member
                elif ext in ['.txt', '.caption']:
                    samples[key]['txt'] = member
                elif ext == '.json':
                    samples[key]['json'] = member

            # Фильтруем только полные пары (картинка + текст)
            valid_keys = [k for k, v in samples.items() if 'img' in v and ('txt' in v or 'json' in v)]
            
            if not valid_keys:
                print("❌ No valid image-text pairs found in shard.")
                return

            # Выбираем случайные
            chosen_keys = random.sample(valid_keys, min(num_samples, len(valid_keys)))
            
            for i, key in enumerate(chosen_keys):
                entry = samples[key]
                
                # Читаем картинку
                img_file = tar.extractfile(entry['img'])
                image = Image.open(io.BytesIO(img_file.read()))
                
                # Читаем текст
                caption = "N/A"
                if 'txt' in entry:
                    txt_file = tar.extractfile(entry['txt'])
                    caption = txt_file.read().decode('utf-8').strip()
                elif 'json' in entry:
                    import json
                    json_file = tar.extractfile(entry['json'])
                    data = json.load(json_file)
                    caption = data.get('caption', data.get('text', 'N/A'))

                # Сохраняем для просмотра
                save_name = f"{name}_sample_{i}_{key}.jpg"
                save_path = output_path / save_name
                image.save(save_path)
                
                print(f"\n🔍 Sample {i+1} (Key: {key})")
                print(f"   🖼️ Image saved to: {save_path}")
                print(f"   📝 Caption: \"{caption}\"")
                print(f"   📏 Image size: {image.size}")

    except Exception as e:
        print(f"Error processing {random_tar}: {e}")

def main():
    base_dir = "/home/jovyan/vasiliev/notebooks/Show-o"
    output_dir = os.path.join(base_dir, "output/debug_data_inspection")
    
    # CC12M
    inspect_dataset(
        "CC12M", 
        os.path.join(base_dir, "data/cc12m"), 
        output_dir
    )
    
    # LAION
    inspect_dataset(
        "LAION", 
        os.path.join(base_dir, "data/laion10k"), 
        output_dir
    )

if __name__ == "__main__":
    main()







