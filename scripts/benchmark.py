"""Model size, compute, CPU latency and peak memory. Each variant runs in a fresh process so
peak working set is not polluted by the others.

  python scripts/benchmark.py            -> results/efficiency.json
"""
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

TILE = 256  # packed side, i.e. a 512 x 512 raw tile = 0.26 MP
THREADS = 2  # matches the free hosting tier


def worker(path):
    import onnxruntime as ort
    import psutil
    so = ort.SessionOptions()
    so.intra_op_num_threads = THREADS
    proc = psutil.Process()
    base = proc.memory_info().rss
    s = ort.InferenceSession(path, so, providers=["CPUExecutionProvider"])
    x = np.random.rand(1, 6, TILE, TILE).astype(np.float32)
    for _ in range(3):
        s.run(None, {"x": x})
    ts = []
    for _ in range(15):
        t = time.perf_counter()
        s.run(None, {"x": x})
        ts.append(time.perf_counter() - t)
    peak = getattr(proc.memory_info(), "peak_wset", proc.memory_info().rss)
    print(json.dumps(dict(ms_per_tile=float(np.median(ts) * 1000), peak_mb=peak / 2**20, runtime_delta_mb=(peak - base) / 2**20)))


def complexity(arch, in_ch=6):
    import torch
    from isp import models as M
    net = M.build(arch, in_ch=in_ch)
    macs = []

    def hook(m, _, y):
        macs.append(y.numel() * m.in_channels // m.groups * m.kernel_size[0] * m.kernel_size[1])

    for m in net.modules():
        if isinstance(m, torch.nn.Conv2d):
            m.register_forward_hook(hook)
    with torch.no_grad():
        net(torch.zeros(1, in_ch, TILE, TILE))
    # A tile is 0.262 MP of raw pixels.
    return sum(p.numel() for p in net.parameters()), sum(macs) / 1e9 / (4 * TILE * TILE / 1e6)


def classical_latency():
    from isp import pipeline as P
    from isp.inputs import CAMS
    import cv2
    cv2.setNumThreads(THREADS)
    b = np.random.rand(2 * TILE, 2 * TILE).astype(np.float32) * .3
    s = P.Scene("rggb", CAMS["IP"]["color_matrix"], [.5, 1, .5], 1e-3, 1e-5)
    P.classical_isp(b, s)
    t = time.perf_counter()
    for _ in range(5):
        P.classical_isp(b, s)
    return (time.perf_counter() - t) / 5 * 1000


def teacher_onnx():
    import torch
    from isp import models as M
    net = M.build("teacher").eval()
    net.load_state_dict(torch.load("weights/teacher.pt", map_location="cpu", weights_only=True))
    net.pad_to = 1
    torch.onnx.export(net, torch.zeros(1, 6, 128, 128), "weights/teacher_fp32.onnx", input_names=["x"], output_names=["y"],
                      dynamic_axes={"x": {2: "h", 3: "w"}, "y": {2: "h", 3: "w"}}, opset_version=17, dynamo=False)


def main():
    if len(sys.argv) > 2 and sys.argv[1] == "--worker":
        return worker(sys.argv[2])
    if not Path("weights/teacher_fp32.onnx").exists():
        teacher_onnx()
    rows = {}
    for name, f, arch in [("Hybrid teacher FP32", "teacher_fp32", "teacher"), ("Hybrid student FP32", "student_fp32", "student"),
                          ("Hybrid student FP16", "student_fp16", "student"), ("Hybrid student INT8", "student_int8", "student"),
                          ("Light student INT8", "student_s_int8", "student_s")]:
        path = f"weights/{f}.onnx"
        out = subprocess.run([sys.executable, __file__, "--worker", path], capture_output=True, text=True, check=True)
        r = json.loads(out.stdout.strip().splitlines()[-1])
        params, gmacs = complexity(arch)
        r.update(params=params, gmacs_per_mp=gmacs, size_mib=Path(path).stat().st_size / 2**20,
                 sec_per_12mp=r["ms_per_tile"] / 1000 * 12 / (4 * TILE * TILE / 1e6))
        rows[name] = r
        print(name, r, flush=True)
    c = classical_latency()
    rows["Classical ISP"] = dict(ms_per_tile=c, sec_per_12mp=c / 1000 * 12 / (4 * TILE * TILE / 1e6), params=0, size_mib=0.0)
    rows["_threads"] = THREADS
    Path("results/efficiency.json").write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main()
