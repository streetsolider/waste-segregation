"""Module 6 - edge optimisation: ONNX export, INT8 quantisation, pruning, latency.

Run:  python -m wasteseg.optimise
For each model: export FP32 ONNX, quantise to INT8 (static, calibrated on 200
training images), then measure file size, single-thread CPU latency and test
accuracy of both. Also measures accuracy after global magnitude pruning.
Writes runs/<model>/optimise.json.
"""
import copy
import json
import time

import numpy as np
import onnxruntime as ort
import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
from onnxruntime.quantization import (CalibrationDataReader, QuantFormat, QuantType,
                                      quantize_static)
from onnxruntime.quantization.shape_inference import quant_pre_process
from torch.utils.data import DataLoader

from . import config as C
from .data import WasteDataset, eval_tf
from .evaluate import load
from .train import predict

PRUNE_LEVELS = [0.3, 0.5, 0.7]


class Reader(CalibrationDataReader):
    def __init__(self, arrays):
        self.it = iter([{"input": a[None]} for a in arrays])

    def get_next(self):
        return next(self.it, None)


def session(path):
    so = ort.SessionOptions()
    so.intra_op_num_threads = 1          # one core, closer to a small edge CPU
    so.inter_op_num_threads = 1
    return ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])


def onnx_accuracy(sess, x, y):
    pred = [sess.run(None, {"input": x[i:i + 1]})[0].argmax(1)[0] for i in range(len(x))]
    return float((np.array(pred) == y).mean())


def latency_ms(sess, x, runs=200, warm=20):
    one = x[:1]
    for _ in range(warm):
        sess.run(None, {"input": one})
    ts = []
    for i in range(runs):
        a = x[i % len(x):i % len(x) + 1]
        t = time.perf_counter(); sess.run(None, {"input": a}); ts.append((time.perf_counter() - t) * 1e3)
    return float(np.median(ts)), float(np.percentile(ts, 95))


def main():
    test = WasteDataset("test", eval_tf)
    x = np.stack([test[i][0].numpy() for i in range(len(test))]).astype(np.float32)
    y = np.array(test.labels)
    cal_ds = WasteDataset("train", eval_tf)
    rng = np.random.default_rng(C.SEED)
    cal = [cal_ds[int(i)][0].numpy().astype(np.float32) for i in rng.choice(len(cal_ds), 200, replace=False)]

    for name in C.MODELS:
        out = C.RUNS / name
        model = load(name, "cpu")
        fp32, pre, int8 = out / "fp32.onnx", out / "fp32_pre.onnx", out / "int8.onnx"
        torch.onnx.export(model, torch.from_numpy(x[:1]), str(fp32), input_names=["input"],
                          output_names=["logits"], dynamic_axes={"input": {0: "n"}, "logits": {0: "n"}},
                          opset_version=17, dynamo=False)
        quant_pre_process(str(fp32), str(pre))
        quantize_static(str(pre), str(int8), Reader(cal), quant_format=QuantFormat.QDQ,
                        per_channel=True, activation_type=QuantType.QUInt8, weight_type=QuantType.QInt8)
        res = {}
        for tag, path in (("fp32", fp32), ("int8", int8)):
            s = session(path)
            med, p95 = latency_ms(s, x)
            res[tag] = {"size_mb": path.stat().st_size / 2 ** 20, "accuracy": onnx_accuracy(s, x, y),
                        "latency_ms_median": med, "latency_ms_p95": p95}
        # global unstructured magnitude pruning (no fine-tuning afterwards)
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        res["pruned"] = {}
        for amount in PRUNE_LEVELS:
            m = copy.deepcopy(model).to(dev)
            targets = [(mod, "weight") for mod in m.modules() if isinstance(mod, (nn.Conv2d, nn.Linear))]
            prune.global_unstructured(targets, prune.L1Unstructured, amount=amount)
            yt, pt = predict(m, DataLoader(test, 64), dev)
            res["pruned"][str(amount)] = float((yt == pt).mean())
        (out / "optimise.json").write_text(json.dumps(res, indent=1))
        print(name, json.dumps(res))


if __name__ == "__main__":
    main()
