"""Module 8 - smart-bin inference: classify one camera image with the INT8 model.

Run:  python -m wasteseg.predict path/to/photo.jpg [model]
"""
import sys

import numpy as np
import onnxruntime as ort
from PIL import Image

from . import config as C
from .data import eval_tf


def classify(path, name="efficientnet_b0"):
    sess = ort.InferenceSession(str(C.RUNS / name / "int8.onnx"), providers=["CPUExecutionProvider"])
    x = eval_tf(Image.open(path).convert("RGB")).numpy()[None].astype(np.float32)
    logits = sess.run(None, {"input": x})[0][0]
    p = np.exp(logits - logits.max()); p /= p.sum()
    return sorted(zip(C.CLASSES, p.tolist()), key=lambda t: -t[1])


if __name__ == "__main__":
    for cls, prob in classify(*sys.argv[1:3]):
        print(f"{cls:10s} {prob:.3f}")
