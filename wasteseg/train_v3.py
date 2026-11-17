"""Cross-dataset experiments (v3). Results go to runs_v3/ ; runs/ and runs_v2/ are not touched.

Run:  python -m wasteseg.train_v3 transfer   # E1: v2 models (trained on TrashNet only) tested on every external set
      python -m wasteseg.train_v3 multi      # E2: trained on TrashNet + all external training splits, 3 seeds
      python -m wasteseg.train_v3 lodo       # E3: leave one dataset out - train on the rest, test on the unseen set

External sets are prepared by wasteseg.external (mapped to the six classes, cleaned, de-duplicated against
TrashNet, split 70/15/15). The TrashNet test set and its five fixed degradations are the same as in v1 and v2.
"""
import json
import math
import sys
import time

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score, recall_score

from . import config as C
from . import external as E
from . import train_v2 as V2
from .models import build, head_parameters

OUT = C.ROOT / "runs_v3"
DEV = "cuda"
BATCH = 64
EPOCHS_HEAD, EPOCHS_FINE = 1, 14
SEEDS = [42, 43, 44]
HELD_OUT = ["realwaste", "warp", "taco"]     # the three sets photographed in real conditions
torch.backends.cudnn.benchmark = True


def load_ext():
    ext = {}
    for name in E.available():
        ext[name] = {}
        for split in ("train", "val", "test"):
            x, y = E.load(name, split)
            ext[name][split] = (torch.from_numpy(x).permute(0, 3, 1, 2).contiguous(), y)   # uint8, kept on the CPU
    return ext


def centre(x):
    o = (E.SIDE - C.IMG_SIZE) // 2
    return x[:, :, o:o + C.IMG_SIZE, o:o + C.IMG_SIZE]


@torch.no_grad()
def predict_cpu(model, x_cpu):
    """Predict for a uint8 CPU tensor N x 3 x 256 x 256 (central 224 crop), moved to the GPU in chunks."""
    out = [V2.predict(model, centre(x_cpu[i:i + 512]).to(DEV)) for i in range(0, len(x_cpu), 512)]
    return np.concatenate(out) if out else np.zeros(0, dtype=np.int64)


def score(y, p):
    present = sorted(set(y.tolist()))
    rec = recall_score(y, p, labels=present, average=None, zero_division=0)
    return {"n": int(len(y)), "acc": float((p == y).mean()),
            "macro_f1": float(f1_score(y, p, labels=present, average="macro", zero_division=0)),
            "balanced_acc": float(rec.mean()),
            "recall": {C.CLASSES[c]: float(r) for c, r in zip(present, rec)}}


def test_everything(model, tn, ext, res, held_out=None):
    for cond, xs in tn["test_x"].items():
        p = V2.predict(model, xs)
        res[cond] = float((p == tn["test_y"]).mean())
        if cond == "clean":
            res["macro_f1"] = float(f1_score(tn["test_y"], p, average="macro"))
    res["mean_degraded"] = float(np.mean([res[c] for c in C.CORRUPTIONS]))
    res["external"] = {}
    for name, sp in ext.items():
        x, y = sp["test"]
        res["external"][name] = score(y, predict_cpu(model, x))
        if name == held_out:                 # nothing of this set was seen, so all of it is test data
            xa = torch.cat([sp[s][0] for s in ("train", "val", "test")])
            ya = np.concatenate([sp[s][1] for s in ("train", "val", "test")])
            res["held_out_full"] = score(ya, predict_cpu(model, xa))
    return res


# ---------------------------------------------------------------- E1: transfer of the TrashNet-only models
def transfer(tn, ext):
    rows = []
    for name in C.MODELS:
        for tag in ("plain", "robust"):
            for seed in V2.SEEDS:
                model = build(name, pretrained=False).to(DEV).to(memory_format=torch.channels_last)
                model.load_state_dict(torch.load(V2.OUT / name / tag / f"seed{seed}" / "best.pt"))
                r = {"model": name, "robust": tag == "robust", "seed": seed, "external": {}}
                for ds, sp in ext.items():
                    xa = torch.cat([sp[s][0] for s in ("train", "val", "test")])
                    ya = np.concatenate([sp[s][1] for s in ("train", "val", "test")])
                    r["external"][ds] = {"all": score(ya, predict_cpu(model, xa)),
                                         "test": score(sp["test"][1], predict_cpu(model, sp["test"][0]))}
                rows.append(r)
                print(f"transfer {name} {tag} seed {seed} " +
                      " ".join(f"{ds} {v['all']['acc']:.3f}" for ds, v in r["external"].items()), flush=True)
                OUT.mkdir(exist_ok=True)
                (OUT / "transfer.json").write_text(json.dumps(rows, indent=1))


# ---------------------------------------------------------------- E2 / E3: training on several sources
def run(name, lr, seed, sources, tn, ext, save=None, held_out=None, robust=True):
    torch.manual_seed(seed); np.random.seed(seed); torch.cuda.manual_seed_all(seed)
    model = build(name).to(DEV).to(memory_format=torch.channels_last)
    ex = torch.cat([ext[s]["train"][0] for s in sources]) if sources else torch.zeros(0, 3, E.SIDE, E.SIDE,
                                                                                    dtype=torch.uint8)
    ey = torch.from_numpy(np.concatenate([ext[s]["train"][1] for s in sources])) if sources else torch.zeros(0).long()
    n_tn, n = len(tn["train_x"]), len(tn["train_x"]) + len(ex)
    labels = np.concatenate([np.array(tn["train_labels"]), ey.numpy()])
    counts = np.bincount(labels, minlength=len(C.CLASSES))
    weight = torch.tensor(n / (len(C.CLASSES) * counts), device=DEV).float()
    loss_fn = nn.CrossEntropyLoss(weight=weight, label_smoothing=C.LABEL_SMOOTHING)
    scaler = torch.amp.GradScaler("cuda")
    val_x = torch.cat([ext[s]["val"][0] for s in sources]) if sources else None
    val_y = np.concatenate([ext[s]["val"][1] for s in sources]) if sources else None
    steps = n // BATCH
    best, best_state, t0 = -1.0, None, time.time()
    for stage, epochs, stage_lr in (("head", EPOCHS_HEAD, C.LR_HEAD), ("fine", EPOCHS_FINE, lr)):
        for p in model.parameters():
            p.requires_grad = stage == "fine"
        for p in head_parameters(model, name):
            p.requires_grad = True
        opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=stage_lr,
                                weight_decay=C.WEIGHT_DECAY)
        warm = steps * (1 if stage == "fine" else 0)
        total = epochs * steps
        sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda k: (k + 1) / warm if k < warm else
                                                  0.5 * (1 + math.cos(math.pi * (k - warm) / max(1, total - warm))))
        for ep in range(epochs):
            model.train()
            perm = torch.randperm(n)
            for b in range(steps):
                idx = perm[b * BATCH:(b + 1) * BATCH]
                a, e = idx[idx < n_tn], idx[idx >= n_tn] - n_tn
                xs, ys = [], []
                if len(a):                   # TrashNet images are 256 x 341, external ones 256 x 256
                    xs.append(V2.geometric(tn["train_x"][a.to(DEV)].float() / 255)); ys.append(tn["train_y"][a.to(DEV)])
                if len(e):
                    xs.append(V2.geometric(ex[e].to(DEV).float() / 255)); ys.append(ey[e].to(DEV))
                x, y = V2.colour(torch.cat(xs)), torch.cat(ys)
                if robust:
                    x = V2.degrade(x)
                x = ((x - V2.MEAN) / V2.STD).contiguous(memory_format=torch.channels_last)
                opt.zero_grad(set_to_none=True)
                with torch.autocast("cuda", dtype=torch.float16):
                    loss = loss_fn(model(x), y)
                scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); sched.step()
            # model selection: validation data of every source used for training, each source weighted equally
            accs = [float((V2.predict(model, tn["val_x"]) == tn["val_y"]).mean())]
            for s in sources:
                vx, vy = ext[s]["val"]
                accs.append(float((predict_cpu(model, vx) == vy).mean()))
            va = float(np.mean(accs))
            if va > best:
                best = va
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    res = {"model": name, "lr": lr, "seed": seed, "robust": robust, "sources": ["trashnet"] + list(sources),
           "held_out": held_out, "n_train": int(n), "val_acc": best, "train_seconds": time.time() - t0}
    test_everything(model, tn, ext, res, held_out)
    if save is not None:
        save.mkdir(parents=True, exist_ok=True)
        torch.save(best_state, save / "best.pt")
    return res


def line(r):
    return (f"{r['model']} seed {r['seed']} held-out {r['held_out']} val {r['val_acc']:.4f} clean {r['clean']:.4f} "
            f"degraded {r['mean_degraded']:.4f} " + " ".join(f"{d} {v['acc']:.3f}" for d, v in r["external"].items()) +
            (f" | unseen-all {r['held_out_full']['acc']:.3f}" if "held_out_full" in r else "") +
            f" ({r['train_seconds']:.0f}s)")


def multi(tn, ext):
    lrs = json.loads((V2.OUT / "tune.json").read_text())["best_lr"]
    rows = json.loads((OUT / "multi.json").read_text()) if (OUT / "multi.json").exists() else []
    for name in C.MODELS:
        for seed in SEEDS:
            if any(q["model"] == name and q["seed"] == seed for q in rows):
                continue                     # finished before an interruption
            r = run(name, lrs[name], seed, list(ext), tn, ext, save=OUT / name / "multi" / f"seed{seed}")
            rows.append(r)
            print("multi", line(r), flush=True)
            (OUT / "multi.json").write_text(json.dumps(rows, indent=1))


def lodo(tn, ext):
    lrs = json.loads((V2.OUT / "tune.json").read_text())["best_lr"]
    rows = json.loads((OUT / "lodo.json").read_text()) if (OUT / "lodo.json").exists() else []
    for held in [h for h in HELD_OUT if h in ext]:
        for name in C.MODELS:
            if any(q["model"] == name and q["held_out"] == held for q in rows):
                continue
            r = run(name, lrs[name], 42, [s for s in ext if s != held], tn, ext, held_out=held)
            rows.append(r)
            print("lodo", line(r), flush=True)
            (OUT / "lodo.json").write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    tn_data, ext_data = V2.load_data(), load_ext()
    for cmd in sys.argv[1:]:
        {"transfer": transfer, "multi": multi, "lodo": lodo}[cmd](tn_data, ext_data)
