"""Module 1 - dataset collection, cleaning, filtering and splitting.

Run:  python -m wasteseg.data
Reads the raw TrashNet folders, drops unreadable / too-small / duplicate images,
and writes a stratified train/val/test split to data/splits.json.
"""
import hashlib
import json
import random
from collections import Counter, defaultdict

import imagehash
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T

from . import config as C


def clean_and_split():
    report = {"raw": {}, "removed": defaultdict(list)}
    kept = defaultdict(list)
    seen_md5, seen_phash = {}, []
    for cls in C.CLASSES:
        files = sorted((C.RAW_DIR / cls).glob("*.jpg"))
        report["raw"][cls] = len(files)
        for f in files:
            rel = f"{cls}/{f.name}"
            # 1. integrity: the file must decode as an RGB image
            try:
                with Image.open(f) as im:
                    im.verify()
                im = Image.open(f).convert("RGB")
            except Exception:
                report["removed"]["unreadable"].append(rel)
                continue
            # 2. size filter
            if min(im.size) < C.MIN_SIDE:
                report["removed"]["too_small"].append(rel)
                continue
            # 3. exact duplicates (same bytes)
            md5 = hashlib.md5(f.read_bytes()).hexdigest()
            if md5 in seen_md5:
                report["removed"]["exact_duplicate"].append([rel, seen_md5[md5]])
                continue
            # 4. near duplicates (perceptual hash), which would leak across splits
            ph = imagehash.phash(im, hash_size=16)
            dup = next((r for h, r in seen_phash if ph - h <= C.NEAR_DUP_HAMMING), None)
            if dup:
                report["removed"]["near_duplicate"].append([rel, dup])
                continue
            seen_md5[md5] = rel
            seen_phash.append((ph, rel))
            kept[cls].append(rel)

    rng = random.Random(C.SEED)
    splits = {"train": [], "val": [], "test": []}
    for cls, items in kept.items():
        rng.shuffle(items)
        n = len(items)
        n_tr, n_va = round(n * C.SPLIT[0]), round(n * C.SPLIT[1])
        splits["train"] += items[:n_tr]
        splits["val"] += items[n_tr:n_tr + n_va]
        splits["test"] += items[n_tr + n_va:]
    report["kept"] = {c: len(v) for c, v in kept.items()}
    report["split_counts"] = {
        s: dict(Counter(p.split("/")[0] for p in items)) for s, items in splits.items()}
    report["removed_counts"] = {k: len(v) for k, v in report["removed"].items()}
    C.SPLIT_FILE.write_text(json.dumps(splits, indent=1))
    C.CLEAN_REPORT.write_text(json.dumps(report, indent=1))
    return report


# ---------- transforms (pre-processing and augmentation) ----------
normalise = T.Compose([T.ToTensor(), T.Normalize(C.IMAGENET_MEAN, C.IMAGENET_STD)])

train_tf = T.Compose([
    T.RandomResizedCrop(C.IMG_SIZE, scale=(0.6, 1.0)),
    T.RandomHorizontalFlip(),
    T.RandomVerticalFlip(),
    T.RandomRotation(20),
    T.ColorJitter(0.3, 0.3, 0.2, 0.02),
    normalise,
])
eval_resize = T.Compose([T.Resize(256), T.CenterCrop(C.IMG_SIZE)])   # PIL -> PIL
eval_tf = T.Compose([eval_resize, normalise])


class WasteDataset(Dataset):
    """Images are decoded once and kept in memory (the dataset is small)."""

    def __init__(self, split, transform=None, corruption=None):
        paths = json.loads(C.SPLIT_FILE.read_text())[split]
        self.paths = paths
        self.labels = [C.CLASSES.index(p.split("/")[0]) for p in paths]
        self.images = [Image.open(C.RAW_DIR / p).convert("RGB") for p in paths]
        self.transform = transform
        self.corruption = corruption      # callable(PIL, index, dataset) -> PIL, test only

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        im = self.images[i]
        if self.corruption is not None:
            im = self.corruption(eval_resize(im), i, self)
            return normalise(im), self.labels[i]
        return self.transform(im), self.labels[i]


def class_weights(ds):
    """Inverse-frequency weights, to offset the small 'trash' class."""
    counts = Counter(ds.labels)
    w = torch.tensor([len(ds) / (len(C.CLASSES) * counts[i]) for i in range(len(C.CLASSES))])
    return w.float()


if __name__ == "__main__":
    r = clean_and_split()
    print(json.dumps({k: r[k] for k in ("raw", "kept", "removed_counts", "split_counts")}, indent=1))
