"""Module 2 - synthetic deployment conditions for the robustness test.

Each function takes a 224x224 PIL image and returns a degraded copy. All randomness
is seeded by the image index, so every model sees exactly the same corrupted test set.
"""
import random

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

from . import config as C


def occlusion(im, i, ds):
    """A grey block hides 25% of the image area (item partly buried)."""
    rng = random.Random(C.SEED + i)
    im = im.copy()
    w, h = im.size
    bw, bh = w // 2, h // 2
    x, y = rng.randint(0, w - bw), rng.randint(0, h - bh)
    im.paste((110, 110, 110), (x, y, x + bw, y + bh))
    return im


def clutter(im, i, ds):
    """Crops of three other test items are pasted around the border (mixed stream)."""
    rng = random.Random(C.SEED + i)
    im = im.copy()
    w, h = im.size
    s = int(w * 0.36)
    corners = [(0, 0), (w - s, 0), (0, h - s), (w - s, h - s)]
    rng.shuffle(corners)
    for (x, y) in corners[:3]:
        j = rng.randrange(len(ds))
        other = ds.images[j].resize((w, h))
        cx, cy = rng.randint(0, w - s), rng.randint(0, h - s)
        im.paste(other.crop((cx, cy, cx + s, cy + s)), (x, y))
    return im


def contamination(im, i, ds):
    """Dark-brown semi-transparent blotches (food residue, dirt)."""
    rng = np.random.default_rng(C.SEED + i)
    a = np.asarray(im).astype(np.float32)
    h, w, _ = a.shape
    yy, xx = np.mgrid[0:h, 0:w]
    mask = np.zeros((h, w), np.float32)
    for _ in range(12):
        cx, cy, r = rng.integers(0, w), rng.integers(0, h), rng.integers(10, 28)
        mask = np.maximum(mask, np.clip(1 - ((xx - cx) ** 2 + (yy - cy) ** 2) / r ** 2, 0, 1))
    stain = np.array([70, 45, 20], np.float32)
    a = a * (1 - 0.75 * mask[..., None]) + stain * (0.75 * mask[..., None])
    return Image.fromarray(a.clip(0, 255).astype(np.uint8))


def low_light(im, i, ds):
    """Dim, low-contrast light with sensor noise (inside a bin)."""
    rng = np.random.default_rng(C.SEED + i)
    im = ImageEnhance.Contrast(ImageEnhance.Brightness(im).enhance(0.35)).enhance(0.7)
    a = np.asarray(im).astype(np.float32) + rng.normal(0, 8, (im.size[1], im.size[0], 3))
    return Image.fromarray(a.clip(0, 255).astype(np.uint8))


def blur(im, i, ds):
    """Out-of-focus or moving item."""
    return im.filter(ImageFilter.GaussianBlur(3))


FUNCS = {"occlusion": occlusion, "clutter": clutter, "contamination": contamination,
         "low_light": low_light, "blur": blur}
