"""Edge optimisation of the multi-source models (v3): ONNX, INT8 quantisation, pruning, latency.

Run:  python -m wasteseg.optimise_v3
For each architecture (multi-source model, seed 42):
  - FP32 ONNX: size, single-thread CPU latency, accuracy
  - INT8: several static-quantisation settings are tried; the one with the best VALIDATION accuracy is kept
    and then measured on the test data (v1 used a single setting, which broke the two mobile networks)
  - global magnitude pruning at 30 / 50 / 70 % without fine-tuning
Accuracy is reported on the TrashNet test set and on the pooled test splits of the five external sets.
Writes runs_v3/optimise.json.
"""
import copy
import json
import time

import numpy as np
import onnxruntime as ort
import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
from onnxruntime.quantization import CalibrationDataReader, CalibrationMethod, QuantFormat, QuantType, quantize_static
from onnxruntime.quantization.shape_inference import quant_pre_process

from . import config as C
from . import external as E
from .data import WasteDataset, eval_resize
from .models import build

OUT = C.ROOT / "runs_v3"
SOURCES = ["realwaste", "garbage_v2", "recycling_hf", "warp", "taco"]
PRUNE_LEVELS = [0.3, 0.5, 0.7]
MEAN = np.array(C.IMAGENET_MEAN, dtype=np.float32).reshape(1, 3, 1, 1)
STD = np.array(C.IMAGENET_STD, dtype=np.float32).reshape(1, 3, 1, 1)
# name -> quantize_static arguments. "convs only" leaves activations such as hard-swish and the
# squeeze-and-excitation gates in floating point, which is where the mobile networks lose accuracy.
SETTINGS = {
    "minmax, all operators": dict(calibrate_method=CalibrationMethod.MinMax),
    "percentile, all operators": dict(calibrate_method=CalibrationMethod.Percentile,
                                      extra_options={"CalibPercentile": 99.999}),
    "minmax, convolution and linear only": dict(calibrate_method=CalibrationMethod.MinMax,
                                                op_types_to_quantize=["Conv", "Gemm", "MatMul"]),
    "percentile, convolution and linear only": dict(calibrate_method=CalibrationMethod.Percentile,
                                                    extra_options={"CalibPercentile": 99.999},
                                                    op_types_to_quantize=["Conv", "Gemm", "MatMul"]),
}


def prep(x_uint8_nhwc):
    """uint8 N x H x W x 3 (already 224 x 224) -> normalised float32 N x 3 x 224 x 224."""
    return ((x_uint8_nhwc.astype(np.float32) / 255).transpose(0, 3, 1, 2) - MEAN) / STD


def trashnet(split):
    ds = WasteDataset(split)
    return np.stack([np.asarray(eval_resize(im)) for im in ds.images]), np.array(ds.labels)


def external(split, per_source=None, seed=C.SEED):
    xs, ys = [], []
    o = (E.SIDE - C.IMG_SIZE) // 2
    rng = np.random.default_rng(seed)
    for s in SOURCES:
        x, y = E.load(s, split)
        if per_source and len(x) > per_source:
            idx = np.sort(rng.choice(len(x), per_source, replace=False))
            x, y = x[idx], y[idx]
        xs.append(x[:, o:o + C.IMG_SIZE, o:o + C.IMG_SIZE]); ys.append(y)
    return np.concatenate(xs), np.concatenate(ys)


class Reader(CalibrationDataReader):
    def __init__(self, x):
        self.it = iter([{"input": a[None]} for a in x])

    def get_next(self):
        return next(self.it, None)


def session(path):
    so = ort.SessionOptions()
    so.intra_op_num_threads = 1          # one core, closer to a small edge CPU
    so.inter_op_num_threads = 1
    return ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])


def accuracy(sess, x, y):
    pred = np.concatenate([sess.run(None, {"input": x[i:i + 32]})[0].argmax(1) for i in range(0, len(x), 32)])
    return float((pred == y).mean())


def latency_ms(sess, x, runs=200, warm=20):
    for _ in range(warm):
        sess.run(None, {"input": x[:1]})
    ts = []
    for i in range(runs):
        a = x[i % len(x):i % len(x) + 1]
        t = time.perf_counter(); sess.run(None, {"input": a}); ts.append((time.perf_counter() - t) * 1e3)
    return float(np.median(ts)), float(np.percentile(ts, 95))


@torch.no_grad()
def torch_acc(model, x, y, dev):
    pred = np.concatenate([model(torch.from_numpy(x[i:i + 128]).to(dev)).argmax(1).cpu().numpy()
                           for i in range(0, len(x), 128)])
    return float((pred == y).mean())


def main():
    tn_x, tn_y = trashnet("test"); tn_x = prep(tn_x)
    ex_x, ex_y = external("test"); ex_x = prep(ex_x)
    v1, y1 = trashnet("val"); v2, y2 = external("val", per_source=100)
    val_x, val_y = prep(np.concatenate([v1, v2])), np.concatenate([y1, y2])
    c1, _ = trashnet("train"); c2, _ = external("train", per_source=25)
    rng = np.random.default_rng(C.SEED)
    cal = prep(np.concatenate([c1[rng.choice(len(c1), 25, replace=False)], c2]))      # 150 calibration images
    print(f"test: trashnet {len(tn_y)}, external {len(ex_y)}; validation {len(val_y)}; calibration {len(cal)}", flush=True)
    res = {}
    for name in C.MODELS:
        folder = OUT / name / "multi" / "seed42"
        model = build(name, pretrained=False)
        model.load_state_dict(torch.load(folder / "best.pt", map_location="cpu"))
        model.eval()
        fp32, pre = folder / "fp32.onnx", folder / "fp32_pre.onnx"
        torch.onnx.export(model, torch.from_numpy(tn_x[:1]), str(fp32), input_names=["input"],
                          output_names=["logits"], dynamic_axes={"input": {0: "n"}, "logits": {0: "n"}},
                          opset_version=17, dynamo=False)
        quant_pre_process(str(fp32), str(pre))
        s = session(fp32)
        med, p95 = latency_ms(s, tn_x)
        r = {"fp32": {"size_mb": fp32.stat().st_size / 2 ** 20, "trashnet": accuracy(s, tn_x, tn_y),
                      "external": accuracy(s, ex_x, ex_y), "val": accuracy(s, val_x, val_y),
                      "latency_ms_median": med, "latency_ms_p95": p95}, "int8_settings": {}}
        # depthwise convolutions have very different weight ranges per channel and are a known weak point
        import onnx
        depthwise = [n.name for n in onnx.load(str(pre)).graph.node if n.op_type == "Conv"
                     and any(a.name == "group" and a.i > 1 for a in n.attribute)]
        settings = dict(SETTINGS)
        settings["entropy, all operators"] = dict(calibrate_method=CalibrationMethod.Entropy)
        if depthwise:
            settings["percentile, depthwise convolutions left in float"] = dict(
                calibrate_method=CalibrationMethod.Percentile, extra_options={"CalibPercentile": 99.999},
                nodes_to_exclude=depthwise)
            settings["percentile, only ordinary convolutions and linear quantised"] = dict(
                calibrate_method=CalibrationMethod.Percentile, extra_options={"CalibPercentile": 99.999},
                op_types_to_quantize=["Conv", "Gemm", "MatMul"], nodes_to_exclude=depthwise)
        r["depthwise_convolutions"] = len(depthwise)
        for label, kw in settings.items():
            path = folder / f"int8_{len(r['int8_settings'])}.onnx"
            try:
                quantize_static(str(pre), str(path), Reader(cal), quant_format=QuantFormat.QDQ, per_channel=True,
                                activation_type=QuantType.QUInt8, weight_type=QuantType.QInt8, **kw)
                s = session(path)
                med, p95 = latency_ms(s, tn_x)
                r["int8_settings"][label] = {"file": path.name, "size_mb": path.stat().st_size / 2 ** 20,
                                             "val": accuracy(s, val_x, val_y),
                                             "latency_ms_median": med, "latency_ms_p95": p95}
            except Exception as e:                     # a setting that cannot be applied is recorded, not hidden
                r["int8_settings"][label] = {"error": f"{type(e).__name__}: {str(e)[:160]}"}
            print(name, label, {k: (round(v, 4) if isinstance(v, float) else v)
                                for k, v in r["int8_settings"][label].items()}, flush=True)
        ok = {k: v for k, v in r["int8_settings"].items() if "val" in v}
        best = max(ok, key=lambda k: ok[k]["val"])       # chosen on validation data only
        s = session(folder / ok[best]["file"])
        r["int8"] = {"setting": best, **ok[best], "trashnet": accuracy(s, tn_x, tn_y),
                     "external": accuracy(s, ex_x, ex_y)}
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        r["pruned"] = {}
        for amount in PRUNE_LEVELS:
            m = copy.deepcopy(model).to(dev)
            targets = [(mod, "weight") for mod in m.modules() if isinstance(mod, (nn.Conv2d, nn.Linear))]
            prune.global_unstructured(targets, prune.L1Unstructured, amount=amount)
            r["pruned"][str(amount)] = {"trashnet": torch_acc(m, tn_x, tn_y, dev), "external": torch_acc(m, ex_x, ex_y, dev)}
        res[name] = r
        (OUT / "optimise.json").write_text(json.dumps(res, indent=1))
        print(name, "fp32", r["fp32"], "\n  int8", r["int8"], "\n  pruned", r["pruned"], flush=True)


if __name__ == "__main__":
    main()
