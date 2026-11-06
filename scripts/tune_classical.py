"""Pick classical denoiser settings on the held-out scene, with true white balance."""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from isp import data as D, metrics as Me, pipeline as P


def main():
    _, val = D.split(D.load_index())
    tiles = D.val_tiles(val, per_image=4)
    res = {}
    for strength in (1, 2, 3, 4, 6):
        for chroma in (True, False):
            ps = []
            for e, raw, gt in tiles:
                s = P.Scene("rggb", e["cm"], e["neutral"], e["a"], e["b"])
                ps.append(Me.psnr(P.classical_isp(raw, s, P.ClassicalParams(strength=strength, chroma=chroma)), gt))
            res[f"strength={strength} chroma={chroma}"] = float(np.mean(ps))
            print(strength, chroma, round(res[f"strength={strength} chroma={chroma}"], 3), flush=True)
    Path("results/classical_tuning.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
