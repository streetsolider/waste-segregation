"""Improved training recipe (v2): tuned learning rate, longer schedule, optional degradation augmentation.

Run:  python -m wasteseg.train_v2 tune      # learning-rate search, judged on the VALIDATION set only
      python -m wasteseg.train_v2 final     # 3 seeds x {plain, robust} at the chosen learning rate
Results go to runs_v2/ ; the original runs/ folder (v1 results used in the paper) is not touched.

Everything runs on the GPU: images are held as tensors and augmented in batches, so an epoch takes
seconds. The test set and its five fixed degradations are identical to v1, so numbers are comparable.
"""
import json
import math
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score
from torchvision.transforms import functional as TF

from . import config as C
from .corrupt import FUNCS
from .data import WasteDataset, eval_resize
from .models import build, head_parameters

OUT = C.ROOT / "runs_v2"
DEV = "cuda"
EPOCHS_HEAD, EPOCHS_FINE = 3, 37
LRS = [1e-4, 3e-4, 1e-3]
SEEDS = [42, 43, 44]
TRAIN_DEGRADATIONS = ["occlusion", "contamination", "low_light", "blur"]   # clutter is held out (unseen)
P_DEG = 0.25                                # chance that each degradation is applied to a training image
MEAN = torch.tensor(C.IMAGENET_MEAN, device=DEV).view(1, 3, 1, 1)
STD = torch.tensor(C.IMAGENET_STD, device=DEV).view(1, 3, 1, 1)


# ---------------------------------------------------------------- data on the GPU
def to_tensor_stack(images, size=None):
    out = []
    for im in images:
        if size:
            im = im.resize(size)
        out.append(torch.from_numpy(np.asarray(im)).permute(2, 0, 1))
    return torch.stack(out).to(DEV)                      # uint8, N x 3 x H x W


def load_data():
    tr, va, te = WasteDataset("train"), WasteDataset("val"), WasteDataset("test")
    data = {"train_x": to_tensor_stack(tr.images, (341, 256)), "train_y": torch.tensor(tr.labels, device=DEV),
            "val_x": to_tensor_stack([eval_resize(i) for i in va.images]), "val_y": np.array(va.labels),
            "test_y": np.array(te.labels), "train_labels": tr.labels}
    base = [eval_resize(i) for i in te.images]
    data["test_x"] = {"clean": to_tensor_stack(base)}
    for name in C.CORRUPTIONS:                           # the same fixed degraded test images as v1
        data["test_x"][name] = to_tensor_stack([FUNCS[name](im, i, te) for i, im in enumerate(base)])
    return data


def norm(x_uint8):
    return ((x_uint8.float() / 255) - MEAN) / STD


# ---------------------------------------------------------------- augmentation (batched, on GPU)
def geometric(x, size=C.IMG_SIZE):
    """Random resized crop (60-100% of area), rotation up to 20 degrees, horizontal and vertical flips."""
    n, _, h, w = x.shape
    area = torch.empty(n, device=DEV).uniform_(0.6, 1.0)
    ratio = torch.exp(torch.empty(n, device=DEV).uniform_(math.log(3 / 4), math.log(4 / 3))) * h / w
    sx, sy = (area * ratio).sqrt().clamp(max=1), (area / ratio).sqrt().clamp(max=1)
    ang = torch.empty(n, device=DEV).uniform_(-20, 20) * math.pi / 180
    c, s = ang.cos(), ang.sin()
    fx = torch.where(torch.rand(n, device=DEV) < 0.5, -1.0, 1.0)
    fy = torch.where(torch.rand(n, device=DEV) < 0.5, -1.0, 1.0)
    tx = (torch.rand(n, device=DEV) * 2 - 1) * (1 - sx)
    ty = (torch.rand(n, device=DEV) * 2 - 1) * (1 - sy)
    theta = torch.stack([torch.stack([c * sx * fx, -s * sy * h / w * fy, tx], 1),
                         torch.stack([s * sx * w / h * fx, c * sy * fy, ty], 1)], 1)
    grid = F.affine_grid(theta, (n, 3, size, size), align_corners=False)
    return F.grid_sample(x, grid, mode="bilinear", padding_mode="reflection", align_corners=False)


def colour(x):
    n = x.shape[0]
    u = lambda lo, hi: torch.empty(n, 1, 1, 1, device=DEV).uniform_(lo, hi)
    x = x * u(0.7, 1.3)
    m = x.mean((1, 2, 3), keepdim=True)
    x = (x - m) * u(0.7, 1.3) + m
    g = x.mean(1, keepdim=True)
    return ((x - g) * u(0.8, 1.2) + g).clamp(0, 1)


def degrade(x):
    """Randomised versions of four deployment conditions. Strengths are drawn at random, so these are
    not copies of the fixed test degradations."""
    n, _, h, w = x.shape
    pick = lambda: (torch.rand(n, device=DEV) < P_DEG).nonzero().flatten()
    yy, xx = torch.meshgrid(torch.arange(h, device=DEV), torch.arange(w, device=DEV), indexing="ij")
    idx = pick()                                         # contamination: brown stains
    if len(idx):
        k = 14
        cx = torch.randint(0, w, (len(idx), k, 1, 1), device=DEV)
        cy = torch.randint(0, h, (len(idx), k, 1, 1), device=DEV)
        r = torch.randint(8, 30, (len(idx), k, 1, 1), device=DEV).float()
        on = (torch.rand(len(idx), k, 1, 1, device=DEV) < 0.7).float()
        mask = ((1 - ((xx - cx) ** 2 + (yy - cy) ** 2) / r ** 2).clamp(0, 1) * on).amax(1, keepdim=True)
        mask = mask * torch.empty(len(idx), 1, 1, 1, device=DEV).uniform_(0.5, 0.85)
        stain = torch.tensor([70, 45, 20], device=DEV).view(1, 3, 1, 1) / 255
        stain = (stain * torch.empty(len(idx), 3, 1, 1, device=DEV).uniform_(0.6, 1.5)).clamp(0, 1)
        x[idx] = x[idx] * (1 - mask) + stain * mask
    idx = pick()                                         # occlusion: grey block over 10-35% of the image
    for i in idx.tolist():
        a = np.random.uniform(0.10, 0.35) * h * w
        bw = int(min(w, max(8, math.sqrt(a * np.random.uniform(0.5, 2)))))
        bh = int(min(h, max(8, a / bw)))
        x0, y0 = np.random.randint(0, w - bw + 1), np.random.randint(0, h - bh + 1)
        x[i, :, y0:y0 + bh, x0:x0 + bw] = np.random.uniform(0.2, 0.7)
    idx = pick()                                         # blur
    if len(idx):
        x[idx] = TF.gaussian_blur(x[idx], kernel_size=13, sigma=float(np.random.uniform(0.6, 2.6)))
    idx = pick()                                         # low light with sensor noise
    if len(idx):
        u = lambda lo, hi: torch.empty(len(idx), 1, 1, 1, device=DEV).uniform_(lo, hi)
        d = x[idx] * u(0.25, 0.8)
        m = d.mean((1, 2, 3), keepdim=True)
        d = (d - m) * u(0.6, 1.0) + m
        x[idx] = d + torch.randn_like(d) * u(0, 10 / 255)
    return x.clamp(0, 1)


# ---------------------------------------------------------------- training and evaluation
@torch.no_grad()
def predict(model, x_uint8):
    model.eval()
    out = []
    for i in range(0, len(x_uint8), 128):
        with torch.autocast("cuda", dtype=torch.float16):
            out.append(model(norm(x_uint8[i:i + 128]).contiguous(memory_format=torch.channels_last)).argmax(1))
    return torch.cat(out).cpu().numpy()


def run(name, lr, seed, robust, data, save=None):
    torch.manual_seed(seed); np.random.seed(seed); torch.cuda.manual_seed_all(seed)
    model = build(name).to(DEV).to(memory_format=torch.channels_last)
    counts = np.bincount(data["train_labels"], minlength=len(C.CLASSES))
    weight = torch.tensor(len(data["train_labels"]) / (len(C.CLASSES) * counts), device=DEV).float()
    loss_fn = nn.CrossEntropyLoss(weight=weight, label_smoothing=C.LABEL_SMOOTHING)
    scaler = torch.amp.GradScaler("cuda")
    n = len(data["train_x"])
    steps = n // C.BATCH
    best, best_state, t0 = -1.0, None, time.time()
    for stage, epochs, stage_lr in (("head", EPOCHS_HEAD, C.LR_HEAD), ("fine", EPOCHS_FINE, lr)):
        for p in model.parameters():
            p.requires_grad = stage == "fine"
        for p in head_parameters(model, name):
            p.requires_grad = True
        opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=stage_lr,
                                weight_decay=C.WEIGHT_DECAY)
        warm = steps * (2 if stage == "fine" else 0)
        total = epochs * steps
        sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda k: (k + 1) / warm if k < warm else
                                                  0.5 * (1 + math.cos(math.pi * (k - warm) / max(1, total - warm))))
        for ep in range(epochs):
            model.train()
            perm = torch.randperm(n, device=DEV)
            for b in range(steps):
                idx = perm[b * C.BATCH:(b + 1) * C.BATCH]
                x = colour(geometric(data["train_x"][idx].float() / 255))
                if robust:
                    x = degrade(x)
                x = ((x - MEAN) / STD).contiguous(memory_format=torch.channels_last)
                opt.zero_grad(set_to_none=True)
                with torch.autocast("cuda", dtype=torch.float16):
                    loss = loss_fn(model(x), data["train_y"][idx])
                scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); sched.step()
            va = float((predict(model, data["val_x"]) == data["val_y"]).mean())
            if va > best:
                best = va
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    res = {"model": name, "lr": lr, "seed": seed, "robust": robust, "val_acc": best,
           "train_seconds": time.time() - t0}
    if save is not None:                                 # test set is only touched for final runs
        for cond, xs in data["test_x"].items():
            p = predict(model, xs)
            res[cond] = float((p == data["test_y"]).mean())
            if cond == "clean":
                res["macro_f1"] = float(f1_score(data["test_y"], p, average="macro"))
        res["mean_degraded"] = float(np.mean([res[c] for c in C.CORRUPTIONS]))
        save.mkdir(parents=True, exist_ok=True)
        torch.save(best_state, save / "best.pt")
    return res


def tune(data):
    rows = []
    for name in C.MODELS:
        for lr in LRS:
            r = run(name, lr, 42, False, data)
            rows.append(r)
            print(f"tune {name} lr {lr:g} val {r['val_acc']:.4f} ({r['train_seconds']:.0f}s)", flush=True)
    best = {m: max((r for r in rows if r["model"] == m), key=lambda r: r["val_acc"])["lr"] for m in C.MODELS}
    OUT.mkdir(exist_ok=True)
    (OUT / "tune.json").write_text(json.dumps({"rows": rows, "best_lr": best}, indent=1))
    print("best lr", best)


def final(data):
    best = json.loads((OUT / "tune.json").read_text())["best_lr"]
    rows = []
    for name in C.MODELS:
        for robust in (False, True):
            for seed in SEEDS:
                tag = "robust" if robust else "plain"
                r = run(name, best[name], seed, robust, data, save=OUT / name / tag / f"seed{seed}")
                rows.append(r)
                print(f"final {name} {tag} seed {seed} val {r['val_acc']:.4f} clean {r['clean']:.4f} "
                      f"degraded {r['mean_degraded']:.4f} " +
                      " ".join(f"{c[:4]} {r[c]:.3f}" for c in C.CORRUPTIONS), flush=True)
                (OUT / "final.json").write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    d = load_data()
    {"tune": tune, "final": final}[sys.argv[1]](d)
