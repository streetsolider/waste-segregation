"""Print a summary of the cross-dataset results in runs_v3/ (mean over seeds, in percent)."""
import json
from pathlib import Path

import numpy as np

R = Path(__file__).resolve().parent.parent / "runs_v3"
MODELS = ["mobilenet_v3_small", "efficientnet_b0", "resnet50"]
m = lambda v: f"{100 * np.mean(v):5.1f}"

tr = json.loads((R / "transfer.json").read_text())
sets = list(tr[0]["external"])
print("TRANSFER (trained on TrashNet only; accuracy on all images of each set)")
print(f"{'model':20}{'kind':8}" + "".join(f"{s[:11]:>12}" for s in sets))
for mod in MODELS:
    for rb in (False, True):
        rows = [r for r in tr if r["model"] == mod and r["robust"] == rb]
        print(f"{mod:20}{'robust' if rb else 'plain':8}" +
              "".join(f"{m([r['external'][s]['all']['acc'] for r in rows]):>12}" for s in sets))

mu = json.loads((R / "multi.json").read_text())
print("\nMULTI (trained on TrashNet + 5 sets; test splits)")
esets = list(mu[0]["external"])
print(f"{'model':20}{'clean':>8}{'degr':>8}" + "".join(f"{s[:11]:>12}" for s in esets))
for mod in MODELS:
    rows = [r for r in mu if r["model"] == mod]
    print(f"{mod:20}{m([r['clean'] for r in rows]):>8}{m([r['mean_degraded'] for r in rows]):>8}" +
          "".join(f"{m([r['external'][s]['acc'] for r in rows]):>12}" for s in esets))
print("  same test splits, TrashNet-only robust models:")
for mod in MODELS:
    rows = [r for r in tr if r["model"] == mod and r["robust"]]
    print(f"{mod:20}{'':16}" + "".join(f"{m([r['external'][s]['test']['acc'] for r in rows]):>12}" for s in esets))

lo = json.loads((R / "lodo.json").read_text())
print("\nLEAVE ONE DATASET OUT (accuracy / balanced accuracy on all images of the unseen set)")
for r in lo:
    h = r["held_out_full"]
    base = [q for q in tr if q["model"] == r["model"] and q["robust"]]
    print(f"{r['held_out']:10}{r['model']:20} unseen {100 * h['acc']:5.1f} / {100 * h['balanced_acc']:5.1f}   "
          f"trashnet-only {m([q['external'][r['held_out']]['all']['acc'] for q in base])} / "
          f"{m([q['external'][r['held_out']]['all']['balanced_acc'] for q in base])}")

un = json.loads((R / "unseen.json").read_text())
print(f"\nUNSEEN {un['dataset']}: n={un['n']}, removed {un['removed_copies_of_training_images']}, {un.get('per_class')}")
for mod in MODELS:
    for kind in ("trashnet_plain", "trashnet_robust", "multi"):
        rows = [r for r in un["rows"] if r["model"] == mod and r["training"] == kind]
        print(f"{mod:20}{kind:16} acc {m([r['acc'] for r in rows])}  balanced {m([r['balanced_acc'] for r in rows])}  "
              f"range {100 * min(r['acc'] for r in rows):.1f}-{100 * max(r['acc'] for r in rows):.1f}")
