"""Result figures for the paper (final experiments).

Run:  python -m wasteseg.make_figures_v3
Writes figures/fig7_robustness.png (TrashNet, clean and degraded, with and without degradation augmentation)
and figures/fig8_generalisation.png (accuracy on settings by what the model was trained on).
"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from . import config as C

COL = {"mobilenet_v3_small": "#E69F00", "efficientnet_b0": "#0072B2", "resnet50": "#009E73"}
plt.rcParams.update({"font.family": "serif", "font.size": 8, "axes.spines.top": False,
                     "axes.spines.right": False})
NICE = {"clean": "Clean", "occlusion": "Occlusion", "clutter": "Clutter", "contamination": "Contam.",
        "low_light": "Low light", "blur": "Blur"}
V2, V3 = C.ROOT / "runs_v2", C.ROOT / "runs_v3"


def robustness():
    rows = json.loads((V2 / "final.json").read_text())
    conds = ["clean"] + C.CORRUPTIONS
    fig, axes = plt.subplots(2, 1, figsize=(3.4, 3.7), sharex=True)
    for ax, robust, title in zip(axes, (False, True), ("(a) Standard augmentation",
                                                       "(b) With degradation augmentation")):
        for k, m in enumerate(C.MODELS):
            rs = [r for r in rows if r["model"] == m and r["robust"] == robust]
            mean = np.array([np.mean([r[c] for r in rs]) for c in conds]) * 100
            lo = mean - np.array([min(r[c] for r in rs) for c in conds]) * 100
            hi = np.array([max(r[c] for r in rs) for c in conds]) * 100 - mean
            ax.bar(np.arange(len(conds)) + (k - 1) * 0.26, mean, 0.26, yerr=[lo, hi], color=COL[m],
                   label=C.MODEL_LABELS[m], error_kw={"lw": 0.6, "capsize": 1.2})
        ax.set_ylim(0, 100); ax.set_ylabel("Test accuracy (%)")
        ax.set_title(title, fontsize=8, loc="left", pad=2)
    axes[1].set_xticks(range(len(conds)), [NICE[c] for c in conds], rotation=25, ha="right")
    axes[0].legend(frameon=False, fontsize=7, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.33),
                   columnspacing=0.8, handlelength=1.2)
    fig.tight_layout(pad=0.3)
    fig.savefig(C.FIGS / "fig7_robustness.png", dpi=300)


def generalisation(model="resnet50"):
    tr = [r for r in json.loads((V3 / "transfer.json").read_text()) if r["model"] == model and r["robust"]]
    mu = [r for r in json.loads((V3 / "multi.json").read_text()) if r["model"] == model]
    lo = {r["held_out"]: r for r in json.loads((V3 / "lodo.json").read_text()) if r["model"] == model}
    un = [r for r in json.loads((V3 / "unseen.json").read_text())["rows"] if r["model"] == model]
    sets = [("realwaste", "RealWaste\n(landfill)"), ("warp", "WaRP\n(conveyor)"), ("taco", "TACO\n(street)"),
            ("trashbox", "TrashBox\n(web)")]
    base, other, own = [], [], []
    for ds, _ in sets:
        if ds == "trashbox":
            base.append(np.mean([r["acc"] for r in un if r["training"] == "trashnet_robust"]))
            other.append(np.mean([r["acc"] for r in un if r["training"] == "multi"]))
            own.append(np.nan)
        else:
            base.append(np.mean([r["external"][ds]["all"]["acc"] for r in tr]))
            other.append(lo[ds]["held_out_full"]["acc"])
            own.append(np.mean([r["external"][ds]["acc"] for r in mu]))
    fig, ax = plt.subplots(figsize=(3.4, 2.5))
    x = np.arange(len(sets))
    for k, (vals, lab, col) in enumerate([(base, "TrashNet only", "#999999"),
                                          (other, "Other datasets added", "#56B4E9"),
                                          (own, "Own images added", "#009E73")]):
        v = np.array(vals) * 100
        ax.bar(x + (k - 1) * 0.27, v, 0.27, color=col, label=lab)
        for xi, vi in zip(x + (k - 1) * 0.27, v):
            if not np.isnan(vi):
                ax.text(xi, vi + 1.5, f"{vi:.0f}", ha="center", fontsize=6.5)
    ax.set_xticks(x, [s[1] for s in sets]); ax.set_ylim(0, 108); ax.set_yticks(range(0, 101, 20))
    ax.set_ylabel("Accuracy (%)")
    ax.legend(frameon=False, fontsize=7, ncol=3, loc="upper center", bbox_to_anchor=(0.46, 1.17), columnspacing=0.7,
              handlelength=1.0, handletextpad=0.4)
    fig.tight_layout(pad=0.3)
    fig.savefig(C.FIGS / "fig8_generalisation.png", dpi=300)


if __name__ == "__main__":
    robustness(); generalisation()
    print("figures written")
