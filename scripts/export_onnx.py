"""Export a restorer to ONNX as FP32, FP16 and static INT8 (QDQ, calibrated on real tiles)."""
import argparse
import sys
from pathlib import Path

import numpy as np
import onnx
import torch
from onnxruntime.quantization import CalibrationDataReader, CalibrationMethod, QuantFormat, QuantType, quantize_static
from onnxruntime.quantization.shape_inference import quant_pre_process

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from isp import classical as C, data as D, models as M


class Calib(CalibrationDataReader):
    def __init__(self, tiles, name):
        self.items = iter(tiles)
        self.name = name

    def get_next(self):
        x = next(self.items, None)
        return None if x is None else {self.name: x}


def model_input(e, raw, conditioned=False):
    f = C.pack(raw) / np.array(e["neutral"], np.float32)[[0, 1, 1, 2]]
    f = np.sign(f) * np.sqrt(np.abs(f) + 1e-8)
    planes = np.ones(f.shape[:2] + (2,), np.float32) * ([np.sqrt(e["a"]) * 16, np.sqrt(e["b"]) * 64] if conditioned else [0, 0])
    return np.concatenate([f, planes], -1).transpose(2, 0, 1)[None].astype(np.float32)


def noisier(e, raw, rng, k=3.0):
    """A calibration tile with sensor noise raised, so activation ranges cover hard inputs."""
    var = k * (e["a"] * np.clip(raw, 0, 1) + e["b"]) + 2e-5
    return raw + rng.randn(*raw.shape).astype(np.float32) * np.sqrt(var).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="weights/student.pt")
    ap.add_argument("--arch", default="student")
    ap.add_argument("--out", default="weights/student")
    ap.add_argument("--calib", type=int, default=96)
    ap.add_argument("--conditioned", action="store_true", help="model was trained with noise planes")
    args = ap.parse_args()

    net = M.build(args.arch).eval()
    net.pad_to = 1  # Restorer pads tiles, so the graph needs no dynamic shapes
    net.load_state_dict(torch.load(args.ckpt, map_location="cpu", weights_only=True))
    fp32 = args.out + "_fp32.onnx"
    torch.onnx.export(net, torch.zeros(1, 6, 128, 128), fp32, input_names=["x"], output_names=["y"],
                      dynamic_axes={"x": {2: "h", 3: "w"}, "y": {2: "h", 3: "w"}}, opset_version=17, dynamo=False)

    from onnxruntime.transformers.float16 import convert_float_to_float16
    onnx.save(convert_float_to_float16(onnx.load(fp32), keep_io_types=True), args.out + "_fp16.onnx")

    # Calibrate on training scenes only.
    train, _ = D.split(D.load_index())
    rng = np.random.RandomState(1)
    picks = [train[i] for i in rng.choice(len(train), args.calib, replace=args.calib > len(train))]
    tiles = [model_input(e, r, args.conditioned) for e, r, _ in D.val_tiles(picks, per_image=1, size=256)]
    pre = args.out + "_pre.onnx"
    quant_pre_process(fp32, pre, skip_symbolic_shape=True)
    # Min-max calibration lost 2 dB to activation outliers. Clipping at the 99.9th percentile and
    # keeping the first and last convs in float recovers most of it. 99.5 looks better on clean
    # input but clips noisy activations (scripts/sweep_quant.py scores clean and degraded input).
    quantize_static(pre, args.out + "_int8.onnx", Calib(tiles, "x"), quant_format=QuantFormat.QDQ,
                    activation_type=QuantType.QUInt8, weight_type=QuantType.QInt8, per_channel=True,
                    nodes_to_exclude=["/intro/Conv", "/outro/Conv"], calibrate_method=CalibrationMethod.Percentile,
                    extra_options=dict(ActivationSymmetric=False, CalibPercentile=99.9))
    Path(pre).unlink()
    for k in ("fp32", "fp16", "int8"):
        p = Path(f"{args.out}_{k}.onnx")
        print(p, round(p.stat().st_size / 2**20, 3), "MiB")


if __name__ == "__main__":
    main()
