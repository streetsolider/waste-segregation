"""Module 9 - result figures for the paper.

Run:  python -m wasteseg.make_figures
"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from . import config as C
from .corrupt import FUNCS
from .data import WasteDataset, eval_resize

COL = {"mobilenet_v3_small": "#E69F00", "efficientnet_b0": "#0072B2", "resnet50": "#009E73"}
plt.rcParams.update({"font.family": "serif", "font.size": 8, "axes.spines.top": False,
                     "axes.spines.right": False})
NICE = {"occlusion": "Occlusion", "clutter": "Clutter", "contamination": "Contamination",
        "low_light": "Low light", "blur": "Blur"}


def corruption_samples():
    ds = WasteDataset("test", corruption=lambda im, i, d: im)
    i = next(k for k, p in enumerate(ds.paths) if p.startswith("plastic"))
    base = eval_resize(ds.images[i])
    fig, axes = plt.subplots(2, 3, figsize=(3.4, 2.45))
    for ax, (title, im) in zip(axes.ravel(), [("Clean", base)] + [(NICE[c], FUNCS[c](base, i, ds)) for c in C.CORRUPTIONS]):
        ax.imshow(im); ax.set_title(title, fontsize=8, pad=2); ax.axis("off")
    fig.tight_layout(pad=0.3)
    fig.savefig(C.FIGS / "fig6_corruptions.png", dpi=300)


def robustness(ev):
    conds = ["clean"] + C.CORRUPTIONS
    fig, ax = plt.subplots(figsize=(3.4, 2.3))
    w = 0.26
    for k, m in enumerate(C.MODELS):
        vals = [ev[m]["accuracy"]] + [ev[m]["robust"][c]["accuracy"] for c in C.CORRUPTIONS]
        ax.bar(np.arange(len(conds)) + (k - 1) * w, np.array(vals) * 100, w, label=C.MODEL_LABELS[m], color=COL[m])
    ax.set_xticks(range(len(conds)), ["Clean"] + [NICE[c].replace("Contamination", "Contam.") for c in C.CORRUPTIONS],
                  rotation=25, ha="right")
    ax.set_ylabel("Test accuracy (%)"); ax.set_ylim(0, 100)
    ax.legend(frameon=False, fontsize=7, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.17), columnspacing=0.8,
              handlelength=1.2)
    fig.tight_layout(pad=0.3)
    fig.savefig(C.FIGS / "fig7_robustness.png", dpi=300)


def confusion(ev, m):
    cm = np.array(ev[m]["confusion"])
    fig, ax = plt.subplots(figsize=(3.2, 2.9))
    ax.imshow(cm, cmap="Blues")
    for i in range(6):
        for j in range(6):
            ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=7,
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax.set_xticks(range(6), C.CLASSES, rotation=35, ha="right"); ax.set_yticks(range(6), C.CLASSES)
    ax.set_xlabel("Predicted class"); ax.set_ylabel("True class")
    ax.spines[:].set_visible(False)
    fig.tight_layout(pad=0.3)
    fig.savefig(C.FIGS / "fig8_confusion.png", dpi=300)


def efficiency(op):
    fig, ax = plt.subplots(figsize=(3.4, 2.3))
    for m in C.MODELS:
        a, b = op[m]["fp32"], op[m]["int8"]
        ax.plot([a["latency_ms_median"], b["latency_ms_median"]], [a["accuracy"] * 100, b["accuracy"] * 100],
                color=COL[m], lw=0.8, zorder=1)
        ax.scatter(a["latency_ms_median"], a["accuracy"] * 100, s=28, color=COL[m], label=C.MODEL_LABELS[m], zorder=2)
        ax.scatter(b["latency_ms_median"], b["accuracy"] * 100, s=28, facecolor="white", edgecolor=COL[m], zorder=2)
    ax.set_xscale("log"); ax.set_xticks([1, 2, 5, 10, 20, 50], ["1", "2", "5", "10", "20", "50"]); ax.minorticks_off()
    ax.set_xlabel("Median CPU latency per image (ms, log scale)")
    ax.set_ylabel("Test accuracy (%)")
    h, l = ax.get_legend_handles_labels()
    h += [plt.Line2D([], [], marker="o", ls="", color="grey"),
          plt.Line2D([], [], marker="o", ls="", markerfacecolor="white", color="grey")]
    ax.legend(h, l + ["FP32", "INT8"], frameon=False, fontsize=7)
    fig.tight_layout(pad=0.3)
    fig.savefig(C.FIGS / "fig9_efficiency.png", dpi=300)


if __name__ == "__main__":
    C.FIGS.mkdir(exist_ok=True)
    ev = {m: json.loads((C.RUNS / m / "eval.json").read_text()) for m in C.MODELS}
    op = {m: json.loads((C.RUNS / m / "optimise.json").read_text()) for m in C.MODELS}
    corruption_samples(); robustness(ev)
    confusion(ev, max(C.MODELS, key=lambda m: ev[m]["accuracy"])); efficiency(op)
    print("figures written")
