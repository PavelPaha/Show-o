# Инструкция по загрузке доменных датасетов

## DocVQA (Document Visual Question Answering)

### Способ 1: Через Hugging Face (рекомендуется)

```bash
# Установите datasets если еще не установлен
pip install datasets

# Запустите скрипт
python scripts/download_domain_datasets.py --docvqa
```

Или вручную через Python:

```python
from datasets import load_dataset

# Загрузить датасет
dataset = load_dataset("ashraq/docvqa", split="train")

# Обработать и сохранить в нужном формате
# (см. scripts/download_domain_datasets.py)
```

### Способ 2: Через официальный сайт

1. Перейдите на https://rrc.cvc.uab.es/?ch=17
2. Зарегистрируйтесь или войдите
3. Загрузите датасет (обычно предоставляется как архив)
4. Распакуйте в `./data/docvqa/`
5. Убедитесь, что структура следующая:
   ```
   data/docvqa/
     train.json
     images/
       *.png
   ```

### Формат JSON для DocVQA

```json
[
  {
    "image": "docvqa_000001.png",
    "question": "What is the date?",
    "answers": ["January 1, 2024", "2024-01-01"],
    "answer": "January 1, 2024"
  }
]
```

## Kvasir-VQA (Medical Visual Question Answering)

### Способ 1: Через GitHub

1. Перейдите на https://github.com/simula/kvasir-vqa
2. Следуйте инструкциям в README для загрузки
3. Распакуйте данные в `./data/kvasir/`

### Способ 2: Попробовать через Hugging Face

```bash
python scripts/download_domain_datasets.py --kvasir
```

### Способ 3: Через Zenodo или другие источники

Kvasir-VQA может быть доступен через:
- Zenodo репозитории
- Медицинские датасетные порталы
- Прямые ссылки от авторов

### Формат JSON для Kvasir-VQA

```json
[
  {
    "image": "kvasir_000001.jpg",
    "question": "What type of polyp is visible?",
    "answers": ["Hyperplastic", "Adenomatous"],
    "answer": "Hyperplastic"
  }
]
```

## Автоматическая загрузка

Используйте скрипт для автоматической загрузки:

```bash
# Установите зависимости
pip install datasets tqdm pillow requests

# Загрузить оба датасета
python scripts/download_domain_datasets.py --both

# Или по отдельности
python scripts/download_domain_datasets.py --docvqa
python scripts/download_domain_datasets.py --kvasir

# Создать примеры для тестирования (если реальные данные недоступны)
python scripts/download_domain_datasets.py --create-samples
```

## Создание примеров для тестирования

Если реальные датасеты недоступны, можно создать примеры JSON файлов:

```bash
python scripts/download_domain_datasets.py --create-samples
```

Это создаст:
- `./data/docvqa/train.json` - пример JSON с 10 записями
- `./data/docvqa/images/` - пустые изображения для тестирования
- `./data/kvasir/train.json` - пример JSON с 10 записями
- `./data/kvasir/images/` - пустые изображения для тестирования

## Проверка структуры

После загрузки убедитесь, что структура следующая:

```
data/
├── docvqa/
│   ├── train.json
│   └── images/
│       └── *.png
└── kvasir/
    ├── train.json
    └── images/
        └── *.jpg или *.png
```

## Обновление конфига

После загрузки обновите пути в `configs/showo_mmu_moe.yaml`:

```yaml
dataset:
  params:
    docvqa_data_file_path: "/home/jovyan/vasiliev/notebooks/Show-o/data/docvqa/train.json"
    docvqa_image_root: "/home/jovyan/vasiliev/notebooks/Show-o/data/docvqa/images"
    kvasir_data_file_path: "/home/jovyan/vasiliev/notebooks/Show-o/data/kvasir/train.json"
    kvasir_image_root: "/home/jovyan/vasiliev/notebooks/Show-o/data/kvasir/images"
```

## Полезные ссылки

- **DocVQA**: 
  - Hugging Face: https://huggingface.co/datasets/ashraq/docvqa
  - Официальный сайт: https://rrc.cvc.uab.es/?ch=17
  
- **Kvasir-VQA**:
  - GitHub: https://github.com/simula/kvasir-vqa
  - Paper: Поиск "Kvasir-VQA medical visual question answering"



