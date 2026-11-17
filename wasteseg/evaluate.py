"""Module 5 - accuracy and robustness evaluation on the held-out test set.

Run:  python -m wasteseg.evaluate
Writes runs/<model>/eval.json (clean metrics, confusion matrix, accuracy per corruption).
"""
import json

import torch
from sklearn.metrics import confusion_matrix, f1_score, precision_recall_fscore_support
from torch.utils.data import DataLoader

from . import config as C
from .corrupt import FUNCS
from .data import WasteDataset, eval_tf
from .models import build, n_params
from .train import predict


def load(name, dev):
    m = build(name, pretrained=False)
    m.load_state_dict(torch.load(C.RUNS / name / "best.pt", map_location=dev))
    return m.to(dev).eval()


def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    clean = WasteDataset("test", eval_tf)
    for name in C.MODELS:
        m = load(name, dev)
        y, p = predict(m, DataLoader(clean, 64), dev)
        pr, rc, f1, _ = precision_recall_fscore_support(y, p, labels=range(len(C.CLASSES)), zero_division=0)
        res = {"params": n_params(m), "n_test": len(y),
               "accuracy": float((y == p).mean()),
               "macro_f1": float(f1_score(y, p, average="macro")),
               "per_class": {c: {"precision": float(pr[i]), "recall": float(rc[i]), "f1": float(f1[i])}
                             for i, c in enumerate(C.CLASSES)},
               "confusion": confusion_matrix(y, p).tolist(),
               "wrong": [{"path": clean.paths[i], "true": int(y[i]), "pred": int(p[i])}
                         for i in range(len(y)) if y[i] != p[i]],
               "robust": {}}
        for cname in C.CORRUPTIONS:
            ds = WasteDataset("test", corruption=FUNCS[cname])
            yc, pc = predict(m, DataLoader(ds, 64), dev)
            res["robust"][cname] = {"accuracy": float((yc == pc).mean()),
                                    "macro_f1": float(f1_score(yc, pc, average="macro"))}
        res["robust_mean_accuracy"] = sum(v["accuracy"] for v in res["robust"].values()) / len(C.CORRUPTIONS)
        (C.RUNS / name / "eval.json").write_text(json.dumps(res, indent=1))
        print(name, f"acc {res['accuracy']:.4f} f1 {res['macro_f1']:.4f}",
              {k: round(v['accuracy'], 3) for k, v in res["robust"].items()})


if __name__ == "__main__":
    main()
