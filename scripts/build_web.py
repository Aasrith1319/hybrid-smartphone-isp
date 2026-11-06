"""Copy models, samples and results into docs/ for the GitHub Pages app."""
import json
import shutil
from pathlib import Path

import numpy as np
from PIL import Image

OUT = Path("docs/assets")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for f in ("student_int8.onnx", "student_s_int8.onnx"):
        shutil.copy(Path("weights") / f, OUT / f)
    shutil.copy("isp/cameras.json", OUT / "cameras.json")
    shutil.copy("results/summary.json", OUT / "summary.json")
    index = []
    for p in sorted(Path("samples").glob("*.npz")):
        d = np.load(p)
        bayer = d["bayer"].astype(np.float16)
        (OUT / f"{p.stem}.f16").write_bytes(bayer.tobytes())
        Image.fromarray(d["gt"]).save(OUT / f"{p.stem}.png")
        index.append(dict(name=p.stem.replace("_", " "), file=p.stem, h=int(bayer.shape[0]), w=int(bayer.shape[1]),
                          cfa=str(d["cfa"]), color_matrix=d["color_matrix"].tolist(), neutral=d["neutral"].tolist(),
                          a=float(d["a"]), b=float(d["b"])))
    (OUT / "samples.json").write_text(json.dumps(index, indent=1))
    (Path("docs") / ".nojekyll").write_text("")
    print(len(index), "samples")


if __name__ == "__main__":
    main()
