# AI-Optimised Waste Segregation - code

Code for the UWE Advanced Artificial Intelligence group paper. Each module is one stage of
the pipeline in Figure 3 of the paper.

| Step | Module | Command | Output |
|---|---|---|---|
| 1 Dataset cleaning, filtering, split | `wasteseg/data.py` | `python -m wasteseg.data` | `data/splits.json`, `data/cleaning_report.json` |
| 2 Synthetic deployment conditions | `wasteseg/corrupt.py` | (used by step 5) | - |
| 3 Model architecture | `wasteseg/models.py` | (used by step 4) | - |
| 4 Two-stage transfer learning | `wasteseg/train.py` | `python -m wasteseg.train` | `runs/<model>/best.pt`, `history.json` |
| 5 Accuracy and robustness | `wasteseg/evaluate.py` | `python -m wasteseg.evaluate` | `runs/<model>/eval.json` |
| 6 Quantisation, pruning, latency | `wasteseg/optimise.py` | `python -m wasteseg.optimise` | `runs/<model>/*.onnx`, `optimise.json` |
| 7 Grad-CAM on errors | `wasteseg/gradcam.py` | `python -m wasteseg.gradcam` | `figures/gradcam_errors.png` |
| 8 Smart-bin inference on one photo | `wasteseg/predict.py` | `python -m wasteseg.predict photo.jpg` | class probabilities |
| 9 Result figures | `wasteseg/make_figures.py` | `python -m wasteseg.make_figures` | `figures/fig6..fig9` |

All settings (seed, split, epochs, learning rates) are in `wasteseg/config.py`.

## Setup
1. Python 3.11, `pip install torch torchvision onnx onnxruntime scikit-learn matplotlib pillow imagehash`.
2. Download `dataset-resized.zip` from https://github.com/garythung/trashnet (folder `data/`) and unzip it
   to `data/raw/` so that `data/raw/dataset-resized/<class>/*.jpg` exists.
3. Run steps 1, 4, 5, 6, 7, 9 in order.

## Final experiments (the ones reported in the paper)

| Step | Command | Output |
|---|---|---|
| Clean and split TrashNet | `python -m wasteseg.data` | `data/splits.json`, `data/cleaning_report.json` |
| Collect, map and clean the other datasets | `python -m wasteseg.external` | `data/external_cache/`, `data/external_report.json` |
| Learning-rate search and benchmark training (3 seeds) | `python -m wasteseg.train_v2 tune` then `final` | `runs_v2/` |
| Transfer, multi-source, leave-one-out, unseen test | `python -m wasteseg.train_v3 transfer multi lodo unseen` | `runs_v3/` |
| Quantisation, pruning, latency | `python -m wasteseg.optimise_v3` | `runs_v3/optimise.json` |
| Grad-CAM | `python -m wasteseg.explain_v3` | `runs_v3/gradcam.json`, `figures/gradcam_v3.png` |
| Result figures | `python -m wasteseg.make_figures_v3` | `figures/fig7_robustness.png`, `figures/fig8_generalisation.png` |
| Print every results table | `python tools\summarise_v3.py` | console |

The external datasets are downloaded into `data/external/` (see `wasteseg/external.py` for the folder names).
`train_v3` needs about 17 GB of RAM, so run it on its own. `train.py`, `evaluate.py`, `optimise.py`, `gradcam.py`
and `make_figures.py` are the first-round versions and are no longer used by the paper, except that
`make_figures.py` draws Figure 6.
