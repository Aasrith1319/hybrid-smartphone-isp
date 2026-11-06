"""SIDD-CC benchmark blocks: 29 test instances x 32 blocks, shipped without metadata.

The camera follows from the full-frame shape. White balance and noise are estimated from
the 32 pooled blocks of each scene, standing in for the full frame; ground truth is used
only to report the "oracle" diagnostic rows.
"""
import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import scipy.io as sio

from . import classical as C

CAMS = json.loads((Path(__file__).parent / "cameras.json").read_text())
SHAPE_TO_CAM = {v["shape"]: k for k, v in CAMS.items()}


@dataclass
class TestScene:
    sid: str
    cam: str
    noisy: np.ndarray  # (32, 256, 256)
    gt_raw: np.ndarray
    gt: np.ndarray     # (32, 256, 256, 3) float32


def load(root="data/SIDD-CC"):
    root = Path(root)
    noisy = sio.loadmat(root / "Raw/test/BenchmarkNoisyBlocksRaw.mat")["data"]
    gt_raw = sio.loadmat(root / "Raw/test/BenchmarkGtBlocksRaw.mat")["data"]
    gt = sio.loadmat(root / "SRGB/test/BenchmarkGtBlocksSrgb.mat")["data"]
    rows = [l.split("\t") for l in (root / "Raw/test/test_scenes_info.txt").read_text().splitlines()[1:]]
    return [TestScene(r[0], SHAPE_TO_CAM[r[2]], noisy[i], gt_raw[i], gt[i].astype(np.float32) / 255)
            for i, r in enumerate(rows)]


def pooled_packed(blocks, cfa):
    """All blocks of a scene as one packed canvas, each re-phased to RGGB."""
    return np.concatenate([C.pack(C.to_canonical(b, cfa, pad=2)[0]) for b in blocks], 0)


def half_rgb(packed):
    return cv2.GaussianBlur(C.packed_rgb(packed).astype(np.float32), (0, 0), 1.5)


def oracle_neutral(scene):
    """As-shot neutral recovered from the ground-truth pair. Diagnostics only."""
    cam = CAMS[scene.cam]
    X, Y = [], []
    for b, g in zip(scene.gt_raw, scene.gt):
        canvas, (y0, x0) = C.to_canonical(np.clip(b, 0, 1), cam["cfa"])
        d = C.demosaic_bilinear(canvas)[y0:y0 + 256, x0:x0 + 256]
        X.append(d[8:-8:2, 8:-8:2].reshape(-1, 3))
        Y.append(g[8:-8:2, 8:-8:2].reshape(-1, 3))
    X, Y = np.concatenate(X), np.concatenate(Y)
    ok = (Y.max(1) < .95) & (Y.min(1) > .02)
    fit = np.linalg.lstsq(X[ok], C.srgb_inverse_oetf(Y[ok]), rcond=None)[0].T
    ccm = C.cam_to_srgb(cam["color_matrix"], [1, 1, 1])
    inv = np.array([ccm[:, j] @ fit[:, j] / (ccm[:, j] @ ccm[:, j]) for j in range(3)])
    n = 1 / inv
    return n / n[1]
