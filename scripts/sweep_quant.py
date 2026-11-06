"""Choose the INT8 calibration on the held-out scene (never the test set)."""
import json
import sys
from pathlib import Path

import numpy as np
from onnxruntime.quantization import CalibrationMethod, QuantFormat, QuantType, quantize_static
from onnxruntime.quantization.shape_inference import quant_pre_process

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate import conditions, onnx_core, restorer
from export_onnx import Calib, model_input, noisier
from isp import data as D, metrics as Me, pipeline as P

IO = ["/intro/Conv", "/outro/Conv"]


def main():
    train, val = D.split(D.load_index())
    rng = np.random.RandomState(1)
    picks = [train[i] for i in rng.choice(len(train), 96, replace=False)]
    plain = [model_input(e, r) for e, r, _ in D.val_tiles(picks, per_image=1)]
    mixed = plain[::2] + [model_input(e, noisier(e, r, rng)) for e, r, _ in D.val_tiles(picks[1::2], per_image=1)]
    tiles = D.val_tiles(val, per_image=3)
    conds = {k: v for k, v in conditions().items() if k in ("clean input (as captured)", "extra noise x4", "underexposed -2 EV")}
    pre, out = "weights/_pre.onnx", "weights/_sweep.onnx"
    quant_pre_process("weights/student_fp32.onnx", pre, skip_symbolic_shape=True)

    def score(path):
        r = restorer(onnx_core(path))
        row = {}
        for name, tr in conds.items():
            ps = []
            for i, (e, raw, gt) in enumerate(tiles):
                s = P.Scene("rggb", e["cm"], e["neutral"], e["a"], e["b"])
                if tr:
                    raw, s = tr(raw, s, np.random.RandomState(i))
                ps.append(Me.psnr(P.hybrid_isp(raw, s, r), gt))
            row[name] = float(np.mean(ps))
        row["mean"] = float(np.mean(list(row.values())))
        return row

    res = {"FP32": score("weights/student_fp32.onnx")}
    print("FP32", res["FP32"], flush=True)
    for pct in (99.5, 99.9, 99.95, 99.99):
        for name, calib in (("clean calibration", plain), ("mixed calibration", mixed)):
            quantize_static(pre, out, Calib(calib, "x"), quant_format=QuantFormat.QDQ, activation_type=QuantType.QUInt8,
                            weight_type=QuantType.QInt8, per_channel=True, nodes_to_exclude=IO,
                            calibrate_method=CalibrationMethod.Percentile,
                            extra_options=dict(ActivationSymmetric=False, CalibPercentile=pct))
            key = f"p{pct}, {name}"
            res[key] = score(out)
            print(key, res[key], flush=True)
    Path(pre).unlink()
    Path(out).unlink()
    Path("results/quant_sweep.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
