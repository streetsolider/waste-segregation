"""Shared settings for the waste segregation pipeline."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw" / "dataset-resized"   # TrashNet, one folder per class
SPLIT_FILE = ROOT / "data" / "splits.json"
CLEAN_REPORT = ROOT / "data" / "cleaning_report.json"
RUNS = ROOT / "runs"
FIGS = ROOT / "figures"

CLASSES = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]
MODELS = ["mobilenet_v3_small", "efficientnet_b0", "resnet50"]
MODEL_LABELS = {"mobilenet_v3_small": "MobileNetV3-Small",
                "efficientnet_b0": "EfficientNet-B0",
                "resnet50": "ResNet-50"}

SEED = 42
IMG_SIZE = 224
SPLIT = (0.70, 0.15, 0.15)          # train / validation / test
MIN_SIDE = 128                      # filter: drop images smaller than this
NEAR_DUP_HAMMING = 2                # filter: perceptual-hash distance treated as duplicate

BATCH = 32
EPOCHS_HEAD = 3                     # stage 1: classifier head only
EPOCHS_FINE = 17                    # stage 2: whole network
LR_HEAD = 1e-3
LR_FINE = 1e-4
WEIGHT_DECAY = 1e-4
LABEL_SMOOTHING = 0.1

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

CORRUPTIONS = ["occlusion", "clutter", "contamination", "low_light", "blur"]
