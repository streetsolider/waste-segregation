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
