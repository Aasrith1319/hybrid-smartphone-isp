"""Auto white balance: classical illuminant statistics corrected by a small ridge regression.

Statistics-based estimators are biased per sensor and per scene. A linear model on their
log-chromaticities (the "corrected moments" idea of Finlayson 2013) removes the bias with a
few hundred parameters, which suits the small number of labelled scenes in SIDD.
"""
import json

import numpy as np

from . import classical as C

CAMERAS = ["GP", "IP", "S6", "N6", "G4"]


def _chroma(n):
    n = np.maximum(np.asarray(n, np.float64), 1e-6)
    return np.array([np.log(n[0] / n[1]), np.log(n[2] / n[1])])


def features(rgb, camera=None):
    """`rgb`: half-resolution camera RGB (h, w, 3), ideally lightly denoised."""
    rgb = np.clip(rgb, 0, 1)
    flat = rgb.reshape(-1, 3)
    luma = flat.mean(1)
    bright = flat[luma > np.quantile(luma, .95)].mean(0) + 1e-6
    keep = (flat.max(1) < .95) & (luma > np.quantile(luma, .05))
    flat_ok = flat[keep] if keep.sum() > 100 else flat
    stats = [C.wb_gray_world(flat_ok[None]), C.wb_shades_of_gray(flat_ok[None], 4), bright / bright[1],
             C.wb_gray_edge(rgb.astype(np.float32), 2), C.wb_gray_edge(rgb.astype(np.float32), 6)]
    f = np.concatenate([_chroma(s) for s in stats])
    onehot = np.array([camera == c for c in CAMERAS], np.float64)
    # Per-camera bias plus per-camera scaling of the first statistic.
    return np.concatenate([f, onehot, np.outer(onehot, f[:2]).ravel(), [1.0]])


def angular_error(n1, n2):
    a, b = np.asarray(n1, float), np.asarray(n2, float)
    c = a @ b / (np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.degrees(np.arccos(np.clip(c, -1, 1))))


class RidgeAWB:
    def __init__(self, w=None):
        self.w = w

    def fit(self, X, neutrals, lam=1.0):
        Y = np.stack([_chroma(n) for n in neutrals])
        self.w = np.linalg.solve(X.T @ X + lam * np.eye(X.shape[1]), X.T @ Y)
        return self

    def predict(self, feat):
        y = feat @ self.w
        return np.array([np.exp(y[0]), 1.0, np.exp(y[1])])

    def save(self, path):
        json.dump(self.w.tolist(), open(path, "w"))

    @classmethod
    def load(cls, path):
        return cls(np.array(json.load(open(path))))
