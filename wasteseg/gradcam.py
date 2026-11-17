"""Module 7 - Grad-CAM explanations of misclassified test images.

Run:  python -m wasteseg.gradcam
Saves figures/gradcam_errors.png and runs/gradcam_summary.json.
"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from . import config as C
from .data import eval_resize, normalise
from .evaluate import load
from .models import last_conv


def gradcam(model, layer, x, cls):
    acts, grads = [], []
    h1 = layer.register_forward_hook(lambda m, i, o: acts.append(o))
    h2 = layer.register_full_backward_hook(lambda m, gi, go: grads.append(go[0]))
    logits = model(x)
    model.zero_grad()
    logits[0, cls].backward()
    h1.remove(); h2.remove()
    w = grads[0].mean((2, 3), keepdim=True)
    cam = F.relu((w * acts[0]).sum(1, keepdim=True))
    cam = F.interpolate(cam, x.shape[2:], mode="bilinear", align_corners=False)[0, 0]
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
    return cam.detach().cpu().numpy(), F.softmax(logits, 1)[0].detach().cpu().numpy()


def main(name="resnet50", n=4):
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = load(name, dev)
    layer = last_conv(model, name)
    wrong = json.loads((C.RUNS / name / "eval.json").read_text())["wrong"]
    summary = []
    for w in wrong:
        im = eval_resize(Image.open(C.RAW_DIR / w["path"]).convert("RGB"))
        x = normalise(im)[None].to(dev)
        cam, prob = gradcam(model, layer, x, w["pred"])
        # share of heat-map mass in the central 50% of the frame (object region in TrashNet)
        q = C.IMG_SIZE // 4
        centre = float(cam[q:-q, q:-q].sum() / (cam.sum() + 1e-8))
        summary.append({**w, "confidence": float(prob[w["pred"]]), "true_prob": float(prob[w["true"]]),
                        "centre_mass": centre, "_im": im, "_cam": cam})
    summary.sort(key=lambda s: -s["confidence"])
    show = summary[:n]
    fig, axes = plt.subplots(2, len(show), figsize=(0.85 * len(show), 2.05))
    for k, s in enumerate(show):
        axes[0, k].imshow(s["_im"])
        axes[0, k].set_title(f"true: {C.CLASSES[s['true']]}", fontsize=5.5, pad=2)
        axes[1, k].imshow(s["_im"]); axes[1, k].imshow(s["_cam"], cmap="jet", alpha=0.45)
        axes[1, k].set_title(f"pred: {C.CLASSES[s['pred']]} ({s['confidence']:.2f})", fontsize=5.5, pad=2)
    for a in axes.ravel():
        a.axis("off")
    fig.tight_layout(pad=0.4)
    C.FIGS.mkdir(exist_ok=True)
    fig.savefig(C.FIGS / "gradcam_errors.png", dpi=300)
    for s in summary:
        s.pop("_im"); s.pop("_cam")
    (C.RUNS / "gradcam_summary.json").write_text(json.dumps(
        {"model": name, "n_errors": len(summary),
         "mean_centre_mass": float(np.mean([s["centre_mass"] for s in summary])) if summary else None,
         "errors": summary}, indent=1))
    print("errors", len(summary), "figure saved")


if __name__ == "__main__":
    main()
