# AI-Optimised Waste Segregation

Image classification of household waste into six classes (cardboard, glass, metal, paper, plastic, trash)
with three transfer-learning CNNs: MobileNetV3-Small, EfficientNet-B0 and ResNet-50. The project tests how well
the models hold up on degraded images and on other waste datasets, and how much they lose when they are
compressed for edge devices (INT8 quantisation, pruning).

Code for a UWE Advanced Artificial Intelligence group project.

## Results (accuracy %, mean of 3 seeds)

Trained on TrashNet plus five other datasets:

| Test set | MobileNetV3-S | EfficientNet-B0 | ResNet-50 |
|---|---|---|---|
| TrashNet (clean) | 88.4 | 91.9 | 92.7 |
| TrashNet (degraded) | 81.4 | 85.7 | 87.3 |
| RealWaste | 89.5 | 93.3 | 93.3 |
| Garbage V2 | 90.1 | 92.2 | 93.4 |
| WaRP | 94.3 | 94.9 | 95.6 |
| TACO | 64.5 | 70.7 | 73.2 |
| TrashBox (never trained on) | 65.2 | 69.9 | 74.3 |

Models trained on TrashNet alone reach only 22-58 % on the other datasets. All raw numbers are in `runs_v2/` and
`runs_v3/`. Run `python tools/summarise_v3.py` to print every table.

## Requirements

- Python 3.10 or newer (tested on 3.11), Windows or Linux
- An NVIDIA GPU with CUDA for training (`train_v2`, `train_v3` and `explain_v3` use CUDA directly).
  Results were produced on an RTX 5070 Ti (16 GB). The demo and single-image prediction run on CPU.
- About 17 GB of free RAM for `train_v3`, and about 15 GB of disk for the datasets

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

If `pip` installs a CPU-only PyTorch, install the CUDA build for your system from
https://pytorch.org/get-started/locally/ first, then run the line above again.

## Data

None of the image data is in the repository. Everything goes under `data/`, and all commands are run from the
repository root.

**TrashNet** (main dataset). Download `dataset-resized.zip` from https://github.com/garythung/trashnet (folder
`data/`) and unzip it so that `data/raw/dataset-resized/<class>/*.jpg` exists.

**External datasets** go into `data/external/`:

| Folder | Source | How to get it |
|---|---|---|
| `realwaste_hf/` | Hugging Face `shahzaibvohra/realwaste` | `hf download shahzaibvohra/realwaste --repo-type dataset --local-dir data/external/realwaste_hf` |
| `recycling_hf/` | Hugging Face `viola77data/recycling-dataset` | `hf download viola77data/recycling-dataset --repo-type dataset --local-dir data/external/recycling_hf` |
| `kagglehub/` | Kaggle `sumn2u/garbage-classification-v2` | `KAGGLEHUB_CACHE=data/external/kagglehub python -c "import kagglehub; kagglehub.dataset_download('sumn2u/garbage-classification-v2')"` |
| `warp/` | Kaggle `parohod/warp-waste-recycling-plant-dataset` | download and unzip so that `data/external/warp/Warp-C/` exists |
| `taco/` | https://github.com/pedropro/TACO | clone it and run its `download.py`, then copy `data/annotations.json` and the `batch_*` folders into `data/external/taco/` |
| `trashbox/` | https://github.com/nikhilvenkatkumsetty/TrashBox | clone it into `data/external/trashbox/` (needs `TrashBox_train_dataset_subfolders/`) |

The Kaggle downloads need a Kaggle account and API token (see the kagglehub docs). On Windows PowerShell, set
the cache with `$env:KAGGLEHUB_CACHE = "data/external/kagglehub"` before running the command.
Any dataset that is missing is skipped.

## Running the experiments

Run these in order:

| Step | Command | Output |
|---|---|---|
| 1. Clean and split TrashNet | `python -m wasteseg.data` | `data/splits.json`, `data/cleaning_report.json` |
| 2. Map and clean the other datasets, remove TrashNet copies | `python -m wasteseg.external` | `data/external_cache/`, `data/external_report.json` |
| 3. Learning-rate search, then benchmark training (3 seeds) | `python -m wasteseg.train_v2 tune`, then `python -m wasteseg.train_v2 final` | `runs_v2/` |
| 4. Transfer, multi-source, leave-one-out, unseen test | `python -m wasteseg.train_v3 transfer multi lodo unseen` | `runs_v3/` |
| 5. Quantisation, pruning, latency | `python -m wasteseg.optimise_v3` | `runs_v3/optimise.json` |
| 6. Grad-CAM | `python -m wasteseg.explain_v3` | `runs_v3/gradcam.json`, `figures/gradcam_v3.png` |
| 7. Figures | `python -m wasteseg.make_figures` and `python -m wasteseg.make_figures_v3` | `figures/` |
| 8. Print the result tables | `python tools/summarise_v3.py` | console |

Steps 3 and 4 take several hours on one GPU. Run `train_v3` on its own because of its memory use. The result
JSON files are already in the repository, so step 8 works without retraining.

All settings (seed, split ratios, image size, degradation types) are in `wasteseg/config.py`.

`train.py`, `evaluate.py`, `optimise.py` and `gradcam.py` are the first-round pipeline (single seed, 20 epochs,
results in `runs/`). They still run, but the `_v2`/`_v3` scripts replace them.

## Demo

A small local web app that compares the TrashNet-only model with the multi-dataset model on an uploaded photo.
It needs the weights from steps 3 and 4.

```bash
python -m demo.export     # writes demo/models/*.onnx
python -m demo.app        # then open http://127.0.0.1:8765
```

For a single photo from the command line, run `python -m wasteseg.train` and `python -m wasteseg.optimise` once
(they create `runs/<model>/int8.onnx`), then `python -m wasteseg.predict photo.jpg [model]`.
