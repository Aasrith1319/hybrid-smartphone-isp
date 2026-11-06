"""Find the noise-to-signal level below which the light network is good enough, on the
held-out scene. Prints the PSNR gap between the two networks per noise bin."""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate import onnx_core, restorer
from isp import classical as C, data as D, metrics as Me, pipeline as P


def main():
    _, val = D.split(D.load_index())
    full, light = restorer(onnx_core("weights/student_int8.onnx")), restorer(onnx_core("weights/student_s_int8.onnx"))
    rows = []
    for e, raw, gt in D.val_tiles(val, per_image=6):
        s = P.Scene("rggb", e["cm"], e["neutral"], e["a"], e["b"])
        rel = P.relative_noise(C.pack(raw), e["a"], e["b"])
        rows.append((rel, Me.psnr(P.hybrid_isp(raw, s, full), gt), Me.psnr(P.hybrid_isp(raw, s, light), gt)))
    rows = np.array(sorted(rows))
    out = []
    for chunk in np.array_split(rows, 6):
        out.append(dict(rel_max=float(chunk[-1, 0]), full=float(chunk[:, 1].mean()), light=float(chunk[:, 2].mean())))
        print(out[-1])
    Path("results/route.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
