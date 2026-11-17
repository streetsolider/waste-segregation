"""Module 4 - two-stage transfer learning.

Run:  python -m wasteseg.train [model ...]
Stage 1 trains only the new classifier head on frozen ImageNet features;
stage 2 fine-tunes the whole network at a lower learning rate. The checkpoint
with the best validation accuracy is kept.
"""
import json
import random
import sys
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from . import config as C
from .data import WasteDataset, class_weights, eval_tf, train_tf
from .models import build, head_parameters


def seed_all(seed=C.SEED):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def predict(model, loader, dev):
    model.eval()
    ys, ps = [], []
    for x, y in loader:
        ps.append(model(x.to(dev)).argmax(1).cpu()); ys.append(y)
    return torch.cat(ys).numpy(), torch.cat(ps).numpy()


def train_one(name, train_ds, val_ds, dev):
    seed_all()
    out = C.RUNS / name
    out.mkdir(parents=True, exist_ok=True)
    model = build(name).to(dev)
    tl = DataLoader(train_ds, C.BATCH, shuffle=True, drop_last=True)
    vl = DataLoader(val_ds, 64)
    loss_fn = nn.CrossEntropyLoss(weight=class_weights(train_ds).to(dev),
                                  label_smoothing=C.LABEL_SMOOTHING)
    history, best = [], 0.0
    t0 = time.time()
    for stage, epochs, lr in (("head", C.EPOCHS_HEAD, C.LR_HEAD), ("fine", C.EPOCHS_FINE, C.LR_FINE)):
        for p in model.parameters():
            p.requires_grad = stage == "fine"
        for p in head_parameters(model, name):
            p.requires_grad = True
        opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                lr=lr, weight_decay=C.WEIGHT_DECAY)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs * len(tl))
        for ep in range(epochs):
            model.train()
            tot, correct, seen = 0.0, 0, 0
            for x, y in tl:
                x, y = x.to(dev), y.to(dev)
                opt.zero_grad()
                logits = model(x)
                loss = loss_fn(logits, y)
                loss.backward(); opt.step(); sched.step()
                tot += loss.item() * len(y); seen += len(y)
                correct += (logits.argmax(1) == y).sum().item()
            yv, pv = predict(model, vl, dev)
            va = float((yv == pv).mean())
            history.append({"stage": stage, "epoch": len(history) + 1, "train_loss": tot / seen,
                            "train_acc": correct / seen, "val_acc": va})
            print(f"{name} {stage} {ep + 1}/{epochs} loss {tot / seen:.3f} "
                  f"train {correct / seen:.3f} val {va:.3f}", flush=True)
            if va > best:
                best = va
                torch.save(model.state_dict(), out / "best.pt")
    (out / "history.json").write_text(json.dumps(
        {"history": history, "best_val_acc": best, "train_seconds": time.time() - t0}, indent=1))


if __name__ == "__main__":
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    train_ds, val_ds = WasteDataset("train", train_tf), WasteDataset("val", eval_tf)
    for name in (sys.argv[1:] or C.MODELS):
        train_one(name, train_ds, val_ds, dev)
