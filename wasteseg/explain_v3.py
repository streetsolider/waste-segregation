"""Grad-CAM analysis of the multi-source ResNet-50 (v3) on every test set.

Run:  python -m wasteseg.explain_v3
For each test set: every misclassified image and a sample of correctly classified ones get a Grad-CAM map
for the predicted class. Reported per set: number of errors, mean confidence, and the share of heat-map mass
in the central quarter of the frame, for errors and for correct predictions (so the two can be compared).
Saves figures/gradcam_v3.png (most confident error of each set) and runs_v3/gradcam.json.
"""
import json
from collections import Counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from . import config as C
from . import external as E
from .gradcam import gradcam
from .models import build, last_conv
from .optimise_v3 import OUT, SOURCES, prep, trashnet

LABELS = {"trashnet": "TrashNet", "realwaste": "RealWaste", "garbage_v2": "Garbage V2",
          "recycling_hf": "Recycling", "warp": "WaRP", "taco": "TACO"}
N_CORRECT = 150


def main(name="resnet50"):
    dev = "cuda"
    model = build(name, pretrained=False)
    model.load_state_dict(torch.load(OUT / name / "multi" / "seed42" / "best.pt", map_location="cpu"))
    model = model.to(dev).eval()
    layer = last_conv(model, name)
    o, q = (E.SIDE - C.IMG_SIZE) // 2, C.IMG_SIZE // 4
    sets = {"trashnet": trashnet("test")}
    for s in SOURCES:
        x, y = E.load(s, "test")
        sets[s] = (x[:, o:o + C.IMG_SIZE, o:o + C.IMG_SIZE], y)
    summary, panels = {}, []
    rng = np.random.default_rng(C.SEED)
    for ds, (x, y) in sets.items():
        xf = torch.from_numpy(prep(x))
        with torch.no_grad():
            pred = torch.cat([model(xf[i:i + 128].to(dev)).argmax(1).cpu() for i in range(0, len(xf), 128)]).numpy()
        wrong = np.flatnonzero(pred != y)
        right = np.flatnonzero(pred == y)
        right = rng.choice(right, min(N_CORRECT, len(right)), replace=False)

        def analyse(idx):
            out = []
            for i in idx:
                cam, prob = gradcam(model, layer, xf[i:i + 1].to(dev), int(pred[i]))
                out.append({"i": int(i), "true": int(y[i]), "pred": int(pred[i]), "confidence": float(prob[pred[i]]),
                            "centre_mass": float(cam[q:-q, q:-q].sum() / (cam.sum() + 1e-8)), "_cam": cam})
            return out
        w, r = analyse(wrong), analyse(right)
        pairs = Counter(f"{C.CLASSES[e['true']]} -> {C.CLASSES[e['pred']]}" for e in w).most_common(4)
        summary[ds] = {"n": int(len(y)), "errors": int(len(wrong)), "accuracy": float((pred == y).mean()),
                       "error_confidence": float(np.mean([e["confidence"] for e in w])) if w else None,
                       "correct_confidence": float(np.mean([e["confidence"] for e in r])),
                       "error_centre_mass": float(np.mean([e["centre_mass"] for e in w])) if w else None,
                       "correct_centre_mass": float(np.mean([e["centre_mass"] for e in r])),
                       "top_confusions": pairs}
        if w:
            top = max(w, key=lambda e: e["confidence"])
            panels.append((ds, x[top["i"]], top))
        print(ds, {k: v for k, v in summary[ds].items()}, flush=True)

    cols = 3                                   # two bands of three sets: photo above, heat map below
    bands = -(-len(panels) // cols)
    fig, axes = plt.subplots(2 * bands, cols, figsize=(3.4, 2.5 * bands))
    for k, (ds, im, e) in enumerate(panels):
        top, bot = axes[2 * (k // cols), k % cols], axes[2 * (k // cols) + 1, k % cols]
        top.imshow(im)
        top.set_title(f"{LABELS[ds]}\ntrue: {C.CLASSES[e['true']]}", fontsize=7, pad=2)
        bot.imshow(im); bot.imshow(e["_cam"], cmap="jet", alpha=0.45)
        bot.set_title(f"pred: {C.CLASSES[e['pred']]} ({e['confidence']:.2f})", fontsize=7, pad=2)
    for a in axes.ravel():
        a.axis("off")
    fig.tight_layout(pad=0.4)
    fig.savefig(C.FIGS / "gradcam_v3.png", dpi=300)
    (OUT / "gradcam.json").write_text(json.dumps({"model": name, "sets": summary}, indent=1))
    print("figure saved")


if __name__ == "__main__":
    main()
