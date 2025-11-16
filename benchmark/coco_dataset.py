import os
import random
from torch.utils.data import Dataset
from pycocotools.coco import COCO
from PIL import Image


class COCODataset(Dataset):
    def __init__(self, root, annFile, transform=None):
        self.root = root
        self.annFile = annFile
        self.transform = transform
        self.coco = COCO(annFile)
        self.ids = list(self.coco.imgs.keys())

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        idx = self.ids[idx]
        img_id = self.coco.imgs[idx]["id"]
        img = self.coco.loadImgs(img_id)[0]
        img_path = os.path.join(self.root, img["file_name"])
        img = Image.open(img_path).convert("RGB")
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        anns = self.coco.loadAnns(ann_ids)
        caption = anns[0]["caption"]

        return {
            "caption": caption,
            "image": img
        }

    def restrict_to_subset(self, subset_size, seed=42):
        if subset_size is None or subset_size >= len(self.ids):
            return
        rng = random.Random(seed)
        ids_copy = self.ids[:]
        rng.shuffle(ids_copy)
        self.ids = ids_copy[:subset_size]


if __name__ == "__main__":
    dataset = COCODataset(
        root="/home/jovyan/vasiliev/notebooks/Show-o/train2017",
        annFile="/home/jovyan/vasiliev/notebooks/Show-o/annotations/captions_train2017.json",
    )
    print(len(dataset))
    for i in range(100):
        print(dataset[i])
