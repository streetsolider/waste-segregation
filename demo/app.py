"""Small demo web app: upload or photograph a waste item and compare two models side by side.

Run:  python -m demo.app            (then open http://127.0.0.1:8765)
Uses ONNX Runtime on the CPU only and the Python standard library for the web server.
Images are classified in memory and never written to disk.
"""
import io
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image, ImageOps

HERE = Path(__file__).resolve().parent
CLASSES = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
MAX_BYTES = 12 * 1024 * 1024
PORT = 8765

opts = ort.SessionOptions()
opts.intra_op_num_threads = 2            # leave the CPU to whatever else is running
SESSIONS = {name: ort.InferenceSession(str(HERE / "models" / f"{name}.onnx"), opts,
                                       providers=["CPUExecutionProvider"])
            for name in ("trashnet_only", "all_datasets")}
PAGE = (HERE / "index.html").read_bytes()


def classify(data):
    im = ImageOps.exif_transpose(Image.open(io.BytesIO(data))).convert("RGB")
    k = 256 / min(im.size)
    im = im.resize((max(256, round(im.width * k)), max(256, round(im.height * k))), Image.BILINEAR)
    l, t = (im.width - 224) // 2, (im.height - 224) // 2
    x = np.asarray(im.crop((l, t, l + 224, t + 224)), dtype=np.float32) / 255
    x = ((x - MEAN) / STD).transpose(2, 0, 1)[None]
    out = {}
    for name, s in SESSIONS.items():
        z = s.run(None, {"image": x})[0][0]
        p = np.exp(z - z.max())
        p /= p.sum()
        out[name] = {c: round(float(v), 4) for c, v in zip(CLASSES, p)}
    return out


class Handler(BaseHTTPRequestHandler):
    def _send(self, code, body, ctype):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path.split("?")[0] == "/":
            self._send(200, PAGE, "text/html; charset=utf-8")
        else:
            self._send(404, b"not found", "text/plain")

    def do_POST(self):
        if self.path != "/predict":
            return self._send(404, b"not found", "text/plain")
        n = int(self.headers.get("Content-Length") or 0)
        if not 0 < n <= MAX_BYTES:
            return self._send(413, b'{"error": "image too large (12 MB limit)"}', "application/json")
        try:
            body = json.dumps(classify(self.rfile.read(n))).encode()
        except Exception:
            return self._send(400, b'{"error": "could not read that file as an image"}', "application/json")
        self._send(200, body, "application/json")

    def log_message(self, *args):          # keep the console quiet; nothing about visitors is logged
        pass


if __name__ == "__main__":
    print(f"demo on http://127.0.0.1:{PORT}", flush=True)
    ThreadingHTTPServer(("127.0.0.1", PORT), Handler).serve_forever()
