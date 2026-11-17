"""External datasets: collection, class mapping, cleaning and leakage control.

Run:  python -m wasteseg.external            # index every dataset found under data/external
      python -m wasteseg.external realwaste  # only the named ones

Each dataset is mapped onto the six TrashNet classes (source classes with no counterpart are left out),
cleaned with the same rules as TrashNet (decodable, shorter side >= 128 px, no exact or near duplicates),
and then checked against TrashNet itself: any external image that is a near copy of a TrashNet image is
removed, so that an "external" test is really external and no TrashNet test image leaks into training.
Kept images are cached as 256 x 256 uint8 arrays in data/external_cache/<name>.npy with a .json index.
"""
import hashlib
import json
import random
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor

import imagehash
import numpy as np
from PIL import Image

from . import config as C

EXT = C.ROOT / "data" / "external"
CACHE = C.ROOT / "data" / "external_cache"
SIDE = 256                     # cached size; evaluation takes the central 224 x 224, as for TrashNet
INTRA_DUP = 6                  # 256-bit pHash distance treated as a duplicate inside one dataset
CROSS_DUP = 10                 # looser for copies of TrashNet, which other sets re-compress and resize
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
TACO_MIN_BOX = 96              # TACO objects smaller than this (pixels) are too small to classify
TACO_PAD = 0.15                # context kept around a TACO bounding box


def _by_folder(root, mapping):
    """One folder per source class. mapping: lower-case folder name -> TrashNet class."""
    items = []
    for f in sorted(root.rglob("*")):
        if f.suffix.lower() in IMG_EXT:
            for part in reversed(f.relative_to(root).parts[:-1]):
                if part.lower() in mapping:
                    items.append((str(f), mapping[part.lower()], None, part.lower()))
                    break
    return items


def _realwaste():
    m = {"cardboard": "cardboard", "glass": "glass", "metal": "metal", "paper": "paper", "plastic": "plastic",
         "miscellaneous trash": "trash"}          # left out: food organics, textile trash, vegetation
    root = EXT / "realwaste_hf"
    if root.exists() and any(root.glob("*.jpg")):  # flat mirror: "<Class>_<n>.jpg"
        items = []
        for f in sorted(root.glob("*.jpg")):
            src = f.stem.rsplit("_", 1)[0].lower()
            if src in m:
                items.append((str(f), m[src], None, src))
        return items
    return _by_folder(EXT / "realwaste", m)


def _garbage_v2():
    m = {c: c for c in C.CLASSES}                 # left out: battery, biological, clothes, shoes
    roots = sorted((EXT / "kagglehub").rglob("original"))     # the archive also holds two resized copies
    return _by_folder(roots[-1], m) if roots else []


def _trashbox():
    m = {"cardboard": "cardboard", "glass": "glass", "metal": "metal", "paper": "paper", "plastic": "plastic"}
    m.update({"cigarette butt": "trash", "tetra pak": None})   # left out: e-waste, medical, tetra pak (composite)
    items = _by_folder(EXT / "trashbox" / "TrashBox_train_dataset_subfolders", m)
    return [i for i in items if i[1]]


def _recycling_hf():
    m = {"aluminium": "metal", "cardboard": "cardboard", "glass": "glass", "paper": "paper", "plastic": "plastic",
         "metal": "metal", "hard plastic": "plastic", "soft plastics": "plastic"}
    # left out: batteries, disposable plates, paper towel, polystyrene, takeaway cups (no clear counterpart)
    return _by_folder(EXT / "recycling_hf", m)


def _warp():
    def cls(name):                                # recycling-plant conveyor crops
        n = name.lower()
        if n.startswith("bottle") or n.startswith("canister") or n in ("detergent-color", "detergent-white",
                                                                      "detergent-transparent"):
            return "plastic"
        if n.startswith("glass"):
            return "glass"
        if n.startswith("cans"):
            return "metal"
        return None                               # left out: juice / milk cartons (composite), detergent box
    items = []
    root = EXT / "warp"
    for f in sorted(root.rglob("*")):
        if f.suffix.lower() in IMG_EXT and "warp-c" in str(f).lower():
            c = cls(f.parent.name)
            if c:
                items.append((str(f), c, None, f.parent.name.lower()))
    return items


TACO_MAP = {
    "Aluminium foil": "metal", "Metal bottle cap": "metal", "Food Can": "metal", "Aerosol": "metal",
    "Drink can": "metal", "Metal lid": "metal", "Pop tab": "metal", "Scrap metal": "metal",
    "Glass bottle": "glass", "Broken glass": "glass", "Glass cup": "glass", "Glass jar": "glass",
    "Other plastic bottle": "plastic", "Clear plastic bottle": "plastic", "Plastic bottle cap": "plastic",
    "Disposable plastic cup": "plastic", "Other plastic cup": "plastic", "Plastic lid": "plastic",
    "Other plastic": "plastic", "Plastic film": "plastic", "Six pack rings": "plastic",
    "Garbage bag": "plastic", "Other plastic wrapper": "plastic", "Single-use carrier bag": "plastic",
    "Polypropylene bag": "plastic", "Spread tub": "plastic", "Tupperware": "plastic",
    "Disposable food container": "plastic", "Other plastic container": "plastic", "Plastic glooves": "plastic",
    "Plastic utensils": "plastic", "Squeezable tube": "plastic", "Plastic straw": "plastic",
    "Other carton": "cardboard", "Egg carton": "cardboard", "Corrugated carton": "cardboard",
    "Toilet tube": "cardboard", "Pizza box": "cardboard",
    "Magazine paper": "paper", "Tissues": "paper", "Wrapping paper": "paper", "Normal paper": "paper",
    "Paper bag": "paper", "Paper cup": "paper", "Paper straw": "paper",
    "Cigarette": "trash", "Food waste": "trash", "Shoe": "trash", "Rope & strings": "trash",
    "Unlabeled litter": "trash", "Styrofoam piece": "trash", "Foam cup": "trash", "Foam food container": "trash",
}                                                 # left out: composite items (drink cartons, crisp packets,
                                                  # blister packs, meal cartons), batteries


def _taco():
    ann_file = EXT / "taco" / "annotations.json"
    if not ann_file.exists():
        return []
    ann = json.loads(ann_file.read_text())
    cats = {c["id"]: c["name"] for c in ann["categories"]}
    imgs = {i["id"]: i for i in ann["images"]}
    items = []
    for a in ann["annotations"]:
        c = TACO_MAP.get(cats[a["category_id"]])
        x, y, w, h = a["bbox"]
        f = EXT / "taco" / imgs[a["image_id"]]["file_name"]
        if c and min(w, h) >= TACO_MIN_BOX and f.exists():
            items.append((str(f), c, (x, y, w, h), cats[a["category_id"]].lower()))
    return items


SOURCES = {"realwaste": _realwaste, "garbage_v2": _garbage_v2, "trashbox": _trashbox,
           "recycling_hf": _recycling_hf, "warp": _warp, "taco": _taco}
SETTING = {"realwaste": "landfill, real waste stream", "garbage_v2": "household items, mostly plain background",
           "trashbox": "web images", "recycling_hf": "household recycling photos",
           "warp": "recycling-plant conveyor", "taco": "litter in the wild (cropped objects)"}


def _load(item):
    """Decode, crop, check, hash. Returns None for an unusable image."""
    path, cls, box, src = item
    try:
        raw = open(path, "rb").read()
        with Image.open(path) as im:
            im.verify()
        im = Image.open(path)
        if box is not None:
            from PIL import ImageOps
            im = ImageOps.exif_transpose(im)
        im = im.convert("RGB")
    except Exception:
        return ("unreadable", item)
    key = raw
    if box is not None:                           # square crop around the object, with a little context
        x, y, w, h = box
        s = max(w, h) * (1 + 2 * TACO_PAD)
        cx, cy = x + w / 2, y + h / 2
        l, t = max(0, cx - s / 2), max(0, cy - s / 2)
        r, b = min(im.width, cx + s / 2), min(im.height, cy + s / 2)
        im = im.crop((int(l), int(t), int(r), int(b)))
        key = raw + repr(box).encode()
    if min(im.size) < C.MIN_SIDE:
        return ("too_small", item)
    w, h = im.size
    k = SIDE / min(w, h)
    im = im.resize((max(SIDE, round(w * k)), max(SIDE, round(h * k))), Image.BILINEAR)
    l, t = (im.width - SIDE) // 2, (im.height - SIDE) // 2
    im = im.crop((l, t, l + SIDE, t + SIDE))
    ph = imagehash.phash(im, hash_size=16).hash.flatten()
    return ("ok", item, np.asarray(im, dtype=np.uint8), hashlib.md5(key).hexdigest(), ph)


def _min_dist(a, b, same=False):
    """For each row of bit matrix a: smallest Hamming distance to a row of b, and its index (GPU)."""
    import torch                              # imported here so the decoding worker processes stay light
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    A = torch.from_numpy(a).float().to(dev) * 2 - 1
    B = torch.from_numpy(b).float().to(dev) * 2 - 1
    best, arg = [], []
    for i in range(0, len(A), 4096):
        d = (A.shape[1] - A[i:i + 4096] @ B.T) / 2
        if same:                                  # only earlier rows count, so the first copy is kept
            j = torch.arange(i, min(i + 4096, len(A)), device=dev)
            d = d.masked_fill(torch.arange(len(B), device=dev)[None, :] >= j[:, None], 1e9)
        m = d.min(1)
        best.append(m.values.cpu()); arg.append(m.indices.cpu())
    return torch.cat(best).numpy(), torch.cat(arg).numpy()


def trashnet_hashes():
    out = CACHE / "trashnet_phash.npz"
    if out.exists():
        z = np.load(out, allow_pickle=True)
        return z["bits"], list(z["paths"])
    splits = json.loads(C.SPLIT_FILE.read_text())
    paths = [p for s in ("train", "val", "test") for p in splits[s]]
    bits = []
    for p in paths:
        im = Image.open(C.RAW_DIR / p).convert("RGB")
        k = SIDE / min(im.size)
        im = im.resize((round(im.width * k), round(im.height * k)), Image.BILINEAR)
        l, t = (im.width - SIDE) // 2, (im.height - SIDE) // 2
        bits.append(imagehash.phash(im.crop((l, t, l + SIDE, t + SIDE)), hash_size=16).hash.flatten())
    bits = np.array(bits)
    CACHE.mkdir(parents=True, exist_ok=True)
    np.savez(out, bits=bits, paths=np.array(paths))
    return bits, paths


def index(name):
    items = SOURCES[name]()
    if not items:
        print(name, "not found, skipped")
        return None
    report = {"dataset": name, "setting": SETTING[name], "mapped_files": len(items),
              "source_classes": dict(Counter(i[3] for i in items)), "removed": Counter()}
    with ProcessPoolExecutor(8) as ex:
        res = list(ex.map(_load, items, chunksize=64))
    ok = []
    seen = set()
    for r in res:
        if r[0] != "ok":
            report["removed"][r[0]] += 1
        elif r[3] in seen:
            report["removed"]["exact_duplicate"] += 1
        else:
            seen.add(r[3]); ok.append(r)
    bits = np.array([r[4] for r in ok])
    d, _ = _min_dist(bits, bits, same=True)
    keep = d > INTRA_DUP
    report["removed"]["near_duplicate"] = int((~keep).sum())
    ok = [r for r, k in zip(ok, keep) if k]
    bits = bits[keep]
    tb, tpaths = trashnet_hashes()
    d, j = _min_dist(bits, tb)
    leak = d <= CROSS_DUP
    report["removed"]["copy_of_trashnet"] = int(leak.sum())
    test = set(json.loads(C.SPLIT_FILE.read_text())["test"])
    report["copies_of_trashnet_test"] = int(sum(tpaths[k] in test for k in j[leak]))
    ok = [r for r, k in zip(ok, leak) if not k]

    rng = random.Random(C.SEED)                   # stratified 70 / 15 / 15, like TrashNet
    split = [None] * len(ok)
    for c in C.CLASSES:
        idx = [i for i, r in enumerate(ok) if r[1][1] == c]
        rng.shuffle(idx)
        n_tr, n_va = round(len(idx) * C.SPLIT[0]), round(len(idx) * C.SPLIT[1])
        for k, i in enumerate(idx):
            split[i] = "train" if k < n_tr else "val" if k < n_tr + n_va else "test"
            if name == "warp":                    # video frames: keep the authors' own train / test division
                split[i] = "test" if "test_crops" in ok[i][1][0] else "val" if k % 7 == 0 else "train"
    labels = [C.CLASSES.index(r[1][1]) for r in ok]
    report["kept"] = len(ok)
    report["kept_per_class"] = {c: labels.count(i) for i, c in enumerate(C.CLASSES)}
    report["split_counts"] = dict(Counter(split))
    report["removed"] = dict(report["removed"])
    CACHE.mkdir(parents=True, exist_ok=True)
    np.save(CACHE / f"{name}.npy", np.stack([r[2] for r in ok]))
    (CACHE / f"{name}.json").write_text(json.dumps(
        {"report": report, "labels": labels, "split": split,
         "paths": [str(r[1][0]).replace(str(EXT), "").lstrip("\\/") for r in ok]}, indent=0))
    print(json.dumps(report, indent=1), flush=True)
    return report


def load(name, split=None):
    """Returns (uint8 array N x 256 x 256 x 3, labels) for one external dataset, optionally one split."""
    meta = json.loads((CACHE / f"{name}.json").read_text())
    x = np.load(CACHE / f"{name}.npy", mmap_mode="r")
    y = np.array(meta["labels"])
    if split is None:
        return np.asarray(x), y
    m = np.array(meta["split"]) == split
    return np.asarray(x[m]), y[m]


def available():
    return [n for n in SOURCES if (CACHE / f"{n}.json").exists()]


if __name__ == "__main__":
    names = sys.argv[1:] or list(SOURCES)
    reports = [r for r in (index(n) for n in names) if r]
    allr = {n: json.loads((CACHE / f"{n}.json").read_text())["report"] for n in available()}
    (C.ROOT / "data" / "external_report.json").write_text(json.dumps(allr, indent=1))
