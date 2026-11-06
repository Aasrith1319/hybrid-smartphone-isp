"""Write demo samples from the held-out scene: 768x768 crops with metadata and reference."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from isp import data as D


def main():
    _, val = D.split(D.load_index())
    # One sample per camera, noisiest image first, so the demo shows a clear difference.
    chosen = {}
    for e in sorted(val, key=lambda e: -e["iso"]):
        chosen.setdefault(e["cam"], e)
    Path("samples").mkdir(exist_ok=True)
    for cam, e in chosen.items():
        oy, ox = D._RED[e["cfa"]]
        y, x = (e["h"] // 2 - 384) // 2 * 2 + oy, (e["w"] // 2 - 384) // 2 * 2 + ox
        raw, gt = D.read_crop("data/prep", e, y, x, 768)
        name = f"samples/{cam}_ISO{e['iso']}_{e['temp']}K.npz"
        np.savez_compressed(name, bayer=raw.astype(np.float16), cfa="rggb", color_matrix=np.array(e["cm"]),
                            neutral=np.array(e["neutral"]), camera=cam, a=e["a"], b=e["b"], gt=(gt * 255 + .5).astype(np.uint8))
        print(name)


if __name__ == "__main__":
    main()
