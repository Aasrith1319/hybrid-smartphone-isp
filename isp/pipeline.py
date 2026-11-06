"""The three ISPs behind one interface: classical, learned (end-to-end) and hybrid.

Each takes a linearised Bayer image and a `Scene` (noise, white balance, colour matrix) and
returns sRGB in [0, 1] plus a dict of intermediate stages for the UI.
"""
from dataclasses import dataclass, field

import numpy as np

from . import classical as C


@dataclass
class Scene:
    cfa: str
    color_matrix: np.ndarray
    neutral: np.ndarray
    a: float = 1e-4
    b: float = 1e-6

    @property
    def ccm(self):
        """White-balanced camera RGB to linear sRGB."""
        return C.cam_to_srgb(self.color_matrix, [1, 1, 1])


@dataclass
class Look:
    """Display finishing. The defaults reproduce the neutral reference rendering, which
    has no tone curve or sharpening; `display()` is the phone-like look used in the app."""
    tone: float = 0.0
    sharpen: float = 0.0
    exposure: float = 1.0

    @staticmethod
    def display(scene_noise_rel, exposure=1.0):
        # Sharpen less when noise is high: unsharp masking amplifies residual noise.
        return Look(tone=0.35, sharpen=float(np.clip(0.6 - 6 * scene_noise_rel, 0, 0.6)), exposure=exposure)


def relative_noise(packed, a, b):
    """Noise-to-signal ratio at the median brightness; drives every adaptive decision."""
    m = max(float(np.median(packed)), 1e-3)
    return float(np.sqrt(a * m + b) / m)


@dataclass
class ClassicalParams:
    denoise: bool = True
    chroma: bool = True
    strength: float = 1.0
    demosaic: str = "bilinear"
    extra: dict = field(default_factory=dict)


def finish(lin_wb, scene, look, stages=None):
    """Shared classical back end: CCM, gamma, tone curve, sharpening."""
    srgb_lin = C.apply_matrix(lin_wb * look.exposure, scene.ccm)
    out = C.srgb_oetf(srgb_lin)
    if stages is not None:
        stages["after_ccm_gamma"] = out
    out = C.sharpen(C.tone_map(out, look.tone), look.sharpen)
    return np.clip(out, 0, 1)


def wb_gains(neutral):
    return (1 / np.asarray(neutral, np.float32))[[0, 1, 1, 2]]


def classical_isp(bayer, scene, params=ClassicalParams(), look=Look(), want_stages=False):
    stages = {} if want_stages else None
    canvas, (y0, x0) = C.to_canonical(bayer, scene.cfa)
    h, w = bayer.shape
    packed = C.pack(canvas)
    rel = relative_noise(packed, scene.a, scene.b)
    if params.denoise:
        packed = C.denoise_packed(packed, scene.a, scene.b, params.strength)
    wb = packed * wb_gains(scene.neutral)
    if want_stages:
        stages["denoised_raw"] = C.packed_rgb(np.clip(packed, 0, 1))
        stages["white_balanced"] = C.packed_rgb(np.clip(wb, 0, 1))
    demosaic = C.demosaic_malvar if params.demosaic == "malvar" else C.demosaic_bilinear
    rgb = demosaic(np.ascontiguousarray(C.unpack(wb)))
    if params.chroma:
        # Radius and regularisation scale with the measured noise-to-signal ratio.
        rgb = C.denoise_chroma(rgb, radius=int(np.clip(4 + 40 * rel, 4, 16)), eps=float(np.clip(rel, 1e-3, .3)) ** 2 * .05)
    rgb = np.clip(rgb, 0, 1)
    out = finish(rgb, scene, look, stages)[y0:y0 + h, x0:x0 + w]
    return (out, stages) if want_stages else out


class Restorer:
    """Runs the network core tile by tile through any backend: core(x NCHW) -> 12-channel."""

    def __init__(self, core, tile=256, overlap=16, noise_input=True, conditioned=False):
        """`noise_input`: the core takes the two noise planes at all. `conditioned`: they carry
        real values. Deployed students are trained with zero planes (see docs/DESIGN.md)."""
        self.core, self.tile, self.overlap = core, tile, overlap
        self.noise_input, self.conditioned = noise_input, conditioned

    def __call__(self, wb_packed, a, b):
        """(h, w, 4) white-balanced packed -> (2h, 2w, 3) restored compressed-domain RGB."""
        h, w, _ = wb_packed.shape
        f = np.sign(wb_packed) * np.sqrt(np.abs(wb_packed) + 1e-8)
        x = f
        if self.noise_input:
            planes = np.empty((h, w, 2), np.float32)
            planes[..., 0], planes[..., 1] = (np.sqrt(a) * 16, np.sqrt(b) * 64) if self.conditioned else (0, 0)
            x = np.concatenate([f, planes], -1)
        x = x.transpose(2, 0, 1)[None].astype(np.float32)
        res = np.zeros((2 * h, 2 * w, 3), np.float32)
        step = self.tile - 2 * self.overlap
        for y in range(0, h, step):
            for xx in range(0, w, step):
                ya, xa = max(y - self.overlap, 0), max(xx - self.overlap, 0)
                yb, xb = min(y + step + self.overlap, h), min(xx + step + self.overlap, w)
                # The network needs multiples of 8; pad here so its graph stays conv-only.
                t = x[..., ya:yb, xa:xb]
                th, tw = t.shape[-2:]
                t = np.pad(t, ((0, 0), (0, 0), (0, -th % 8), (0, -tw % 8)), mode="edge")
                out = self.core(np.ascontiguousarray(t))[0][:, :th, :tw]
                out = out.reshape(3, 2, 2, th, tw).transpose(3, 1, 4, 2, 0).reshape(2 * th, 2 * tw, 3)
                ye, xe = min(y + step, h), min(xx + step, w)
                res[2 * y:2 * ye, 2 * xx:2 * xe] = out[2 * (y - ya):2 * (ye - ya), 2 * (xx - xa):2 * (xe - xa)]
        base = C.demosaic_bilinear(np.ascontiguousarray(C.unpack(f)))
        return base + res


def hybrid_isp(bayer, scene, restorer, look=Look(), want_stages=False):
    """Classical front end, learned restoration, classical colour back end."""
    stages = {} if want_stages else None
    canvas, (y0, x0) = C.to_canonical(bayer, scene.cfa)
    h, w = bayer.shape
    wb = C.pack(canvas) * wb_gains(scene.neutral)
    y = restorer(wb, scene.a, scene.b)
    lin = np.clip(np.sign(y) * y * y, 0, 1)
    if want_stages:
        stages["white_balanced"] = C.packed_rgb(np.clip(wb, 0, 1))
        stages["restored_linear"] = lin
    out = finish(lin, scene, look, stages)[y0:y0 + h, x0:x0 + w]
    return (out, stages) if want_stages else out


def learned_isp(bayer, scene, restorer):
    """End-to-end baseline: no white balance, no colour matrix, the network does it all."""
    canvas, (y0, x0) = C.to_canonical(bayer, scene.cfa)
    h, w = bayer.shape
    y = restorer(C.pack(canvas), scene.a, scene.b)
    return C.srgb_oetf(np.sign(y) * y * y)[y0:y0 + h, x0:x0 + w]
