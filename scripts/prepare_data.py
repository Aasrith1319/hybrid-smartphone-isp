"""Convert downloaded SIDD-CC files to memory-mappable arrays and write the index.

Safe to re-run: finished images are skipped, so it can follow a running download.
"""
import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import scipy.io as sio
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from isp import classical as C

Image.MAX_IMAGE_PIXELS = None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="data/SIDD-CC")
    ap.add_argument("--out", default="data/prep")
    args = ap.parse_args()
    src, out = Path(args.src), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    index_path = out / "index.json"
    index = {e["id"]: e for e in json.loads(index_path.read_text())} if index_path.exists() else {}

    for noisy in sorted((src / "Raw/train/Data").glob("*/*_NOISY_RAW_*.MAT")):
        inst = noisy.parent.name
        num, _, _, idx = re.match(r"(\d+)_(NOISY)_(RAW)_(\d+)", noisy.stem).groups()
        key = f"{inst}_{idx}"
        png = src / "SRGB/train/Data" / inst / f"{num}_GT_SRGB_{idx}.PNG"
        if key in index or not png.exists():
            continue
        meta = sio.loadmat(noisy.with_name(f"{num}_METADATA_RAW_{idx}.MAT"))
        raw = sio.loadmat(noisy)["data"]
        gt = np.array(Image.open(png))
        assert raw.shape == gt.shape[:2], key
        cfa = str(meta["cfa_str"][0])
        a, b = C.estimate_noise(C.pack(C.to_canonical(raw, cfa)[0]))
        np.save(out / f"{key}_noisy.npy", raw.astype(np.float16))
        np.save(out / f"{key}_gt.npy", gt)
        p = inst.split("_")
        index[key] = dict(id=key, inst=inst, scene=p[1], cam=p[2], iso=int(p[3]), temp=int(p[5]), light=p[6],
                          cfa=cfa, neutral=meta["AsShotNeutral"].ravel().tolist(),
                          cm=meta["ColorMatrix2"].ravel().tolist(), h=raw.shape[0], w=raw.shape[1], a=a, b=b)
        print(key, f"a={a:.2e} b={b:.2e}", flush=True)
        index_path.write_text(json.dumps(list(index.values()), indent=0))
    print("images:", len(index))


if __name__ == "__main__":
    main()
