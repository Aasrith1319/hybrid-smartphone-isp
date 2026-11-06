"""Training patches and held-out validation tiles."""
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from . import classical as C

VAL_SCENE = "008"  # whole physical scene held out of training, all five cameras
_RED = {"rggb": (0, 0), "grbg": (0, 1), "gbrg": (1, 0), "bggr": (1, 1)}


def load_index(root="data/prep"):
    return json.loads((Path(root) / "index.json").read_text())


def color_correction(entry):
    """Matrix from white-balanced camera RGB to linear sRGB (WB stage removed)."""
    return C.cam_to_srgb(entry["cm"], [1, 1, 1]).astype(np.float32)


def read_crop(root, e, y, x, size):
    raw = np.load(Path(root) / f"{e['id']}_noisy.npy", mmap_mode="r")
    gt = np.load(Path(root) / f"{e['id']}_gt.npy", mmap_mode="r")
    return (np.array(raw[y:y + size, x:x + size], np.float32),
            np.array(gt[y:y + size, x:x + size], np.float32) / 255)


class Patches(Dataset):
    """Random RGGB-phase crops with flips, extra-noise and white-balance jitter."""

    def __init__(self, entries, root="data/prep", size=256, length=16000, augment=True):
        self.e, self.root, self.size, self.length, self.aug = entries, root, size, length, augment

    def __len__(self):
        return self.length

    def __getitem__(self, _):
        rng = np.random
        e = self.e[rng.randint(len(self.e))]
        s = self.size
        oy, ox = _RED[e["cfa"]]
        y = rng.randint((e["h"] - s - 2) // 2) * 2 + oy
        x = rng.randint((e["w"] - s - 2) // 2) * 2 + ox
        raw, gt = read_crop(self.root, e, y, x, s + 1)
        # The crop is odd-sized and starts on red, so it also ends on red: a flip keeps RGGB.
        if rng.rand() < .5:
            raw, gt = raw[:, ::-1], gt[:, ::-1]
        if rng.rand() < .5:
            raw, gt = raw[::-1], gt[::-1]
        raw, gt = np.ascontiguousarray(raw[:s, :s]), np.ascontiguousarray(gt[:s, :s])
        a, b = e["a"], e["b"]
        if self.aug and rng.rand() < .3:
            k = rng.uniform(0, 3)
            raw = raw + rng.randn(*raw.shape).astype(np.float32) * np.sqrt(k * (a * np.clip(raw, 0, 1) + b)).astype(np.float32)
            a, b = a * (1 + k), b * (1 + k)
        jitter = np.exp(rng.uniform(-.12, .12, 3)).astype(np.float32) if self.aug and rng.rand() < .5 else np.ones(3, np.float32)
        gains = jitter / np.array(e["neutral"], np.float32)
        packed = C.pack(raw) * gains[[0, 1, 1, 2]]
        # The renderer must undo the jitter: the target colour does not change.
        ccm = color_correction(e) @ np.diag(1 / jitter).astype(np.float32)
        ccm_raw = np.array(e["neutral"], np.float32)
        return dict(packed=torch.from_numpy(packed.transpose(2, 0, 1).copy()),
                    raw=torch.from_numpy(C.pack(raw).transpose(2, 0, 1).copy()),
                    gt=torch.from_numpy(gt.transpose(2, 0, 1).copy()),
                    ccm=torch.from_numpy(ccm.astype(np.float32)),
                    neutral=torch.from_numpy(ccm_raw),
                    a=torch.tensor(a, dtype=torch.float32), b=torch.tensor(b, dtype=torch.float32))


def split(index):
    train = [e for e in index if e["scene"] != VAL_SCENE]
    val = [e for e in index if e["scene"] == VAL_SCENE]
    return train, val


def val_tiles(val, root="data/prep", size=256, per_image=6, seed=0):
    """Fixed tiles from the held-out scene, with true metadata."""
    rng = np.random.RandomState(seed)
    out = []
    for e in val:
        oy, ox = _RED[e["cfa"]]
        for _ in range(per_image):
            y = rng.randint((e["h"] - size - 2) // 2) * 2 + oy
            x = rng.randint((e["w"] - size - 2) // 2) * 2 + ox
            raw, gt = read_crop(root, e, y, x, size)
            out.append((e, raw, gt))
    return out
