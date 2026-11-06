"""Fit the ridge AWB on training scenes; report held-out error against classical estimators."""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from isp import awb, classical as C, data as D, testset as T

ROOT = Path("data/prep")
_RED = {"rggb": (0, 0), "grbg": (0, 1), "gbrg": (1, 0), "bggr": (1, 1)}


def sample_rgb(e, rng, blocks=32, size=256):
    """Random blocks of one image, matching the statistics of the benchmark's block sets."""
    raw = np.load(ROOT / f"{e['id']}_noisy.npy", mmap_mode="r")
    oy, ox = _RED[e["cfa"]]
    out = []
    for _ in range(blocks):
        y = rng.randint((e["h"] - size) // 2) * 2 + oy
        x = rng.randint((e["w"] - size) // 2) * 2 + ox
        out.append(C.pack(np.array(raw[y:y + size, x:x + size], np.float32)))
    return T.half_rgb(np.concatenate(out, 0))


def main():
    rng = np.random.RandomState(0)
    index = D.load_index()
    train, val = D.split(index)
    feats = {}
    for i, e in enumerate(index):
        feats[e["id"]] = [(lambda r: (awb.features(r, e["cam"]), C.wb_gray_world(r), C.wb_shades_of_gray(r), C.wb_gray_edge(r)))(sample_rgb(e, rng)) for _ in range(4)]
        if i % 40 == 0:
            print(i, flush=True)

    def stack(entries):
        X = np.stack([f[0] for e in entries for f in feats[e["id"]]])
        N = [e["neutral"] for e in entries for _ in feats[e["id"]]]
        return X, N

    # Leave-one-physical-scene-out on the training scenes selects the ridge strength.
    scenes = sorted({e["scene"] for e in train})
    best = None
    for lam in [.01, .03, .1, .3, 1, 3, 10]:
        errs = []
        for s in scenes:
            tr = [e for e in train if e["scene"] != s]
            te = [e for e in train if e["scene"] == s]
            m = awb.RidgeAWB().fit(*stack(tr), lam=lam)
            errs += [awb.angular_error(m.predict(f[0]), e["neutral"]) for e in te for f in feats[e["id"]]]
        print(f"lam {lam} CV mean {np.mean(errs):.2f} deg median {np.median(errs):.2f}", flush=True)
        if best is None or np.mean(errs) < best[0]:
            best = (float(np.mean(errs)), lam)

    model = awb.RidgeAWB().fit(*stack(train), lam=best[1])
    model.save("weights/awb.json")
    report = dict(lam=best[1], cv_mean_deg=best[0])
    for i, name in enumerate(["ridge", "gray_world", "shades_of_gray", "gray_edge"]):
        report[f"val_{name}_deg"] = float(np.mean([
            awb.angular_error(model.predict(f[0]) if i == 0 else f[i], e["neutral"])
            for e in val for f in feats[e["id"]]]))
    print(report)
    Path("results/awb.json").write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
