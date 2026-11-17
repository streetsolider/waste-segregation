"""Export the two demo models to ONNX (CPU only, so a training run on the GPU is not disturbed).

Run:  python -m demo.export
"""
import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""
import torch

from wasteseg import config as C
from wasteseg.models import build

OUT = C.ROOT / "demo" / "models"
MODELS = {   # file name -> (architecture, weights)
    "trashnet_only": ("efficientnet_b0", C.ROOT / "runs_v2" / "efficientnet_b0" / "robust" / "seed42" / "best.pt"),
    "all_datasets": ("efficientnet_b0", C.ROOT / "runs_v3" / "efficientnet_b0" / "multi" / "seed42" / "best.pt"),
}

if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    for name, (arch, weights) in MODELS.items():
        model = build(arch, pretrained=False)
        model.load_state_dict(torch.load(weights, map_location="cpu"))
        model.eval()
        torch.onnx.export(model, torch.zeros(1, 3, C.IMG_SIZE, C.IMG_SIZE), str(OUT / f"{name}.onnx"),
                          input_names=["image"], output_names=["logits"], opset_version=17, dynamo=False)
        print("exported", name)
