"""Classical ISP stages (NumPy / OpenCV only, so the deployed app needs no PyTorch).

Conventions: Bayer images are float32 in [0, 1] after linearisation. "Canonical" Bayer
means RGGB phase. Packed Bayer is (H/2, W/2, 4) in R, G1, G2, B order.
"""
import cv2
import numpy as np

XYZ_TO_SRGB = np.array([[3.2404542, -1.5371385, -0.4985314],
                        [-0.9692660, 1.8760108, 0.0415560],
                        [0.0556434, -0.2040259, 1.0572252]])

# Row/col offset of the red pixel inside the 2x2 CFA tile.
_RED_OFFSET = {"rggb": (0, 0), "grbg": (0, 1), "gbrg": (1, 0), "bggr": (1, 1)}
# Even reflect-padding keeps the CFA phase. The margin must exceed the 8-pixel border that
# training leaves out of the loss, because the network output is unconstrained there.
PAD = 18


def linearize(raw, black, white):
    """Subtract black level and normalise. Sub-black noise is kept (not clipped)."""
    return (raw.astype(np.float32) - black) / (white - black)


def to_canonical(bayer, cfa, pad=PAD):
    """Re-phase any Bayer pattern to RGGB. Returns (canvas, (y0, x0)) with the original
    image at canvas[y0:y0+H, x0:x0+W]; the canvas has even height and width."""
    h, w = bayer.shape
    oy, ox = _RED_OFFSET[cfa.lower()]
    p = np.pad(bayer, pad, mode="reflect")[oy:, ox:]
    hc, wc = (h + 2 * pad - oy) // 2 * 2, (w + 2 * pad - ox) // 2 * 2
    return np.ascontiguousarray(p[:hc, :wc]), (pad - oy, pad - ox)


def pack(rggb):
    return np.stack([rggb[0::2, 0::2], rggb[0::2, 1::2], rggb[1::2, 0::2], rggb[1::2, 1::2]], -1)


def unpack(packed):
    h, w, _ = packed.shape
    out = np.empty((2 * h, 2 * w), packed.dtype)
    out[0::2, 0::2], out[0::2, 1::2] = packed[..., 0], packed[..., 1]
    out[1::2, 0::2], out[1::2, 1::2] = packed[..., 2], packed[..., 3]
    return out


def _masks(h, w):
    r = np.zeros((h, w), np.float32)
    g = r.copy()
    b = r.copy()
    r[0::2, 0::2] = 1
    g[0::2, 1::2] = 1
    g[1::2, 0::2] = 1
    b[1::2, 1::2] = 1
    return r, g, b


def demosaic_bilinear(rggb):
    r, g, b = _masks(*rggb.shape)
    k_g = np.array([[0, 1, 0], [1, 4, 1], [0, 1, 0]], np.float32) / 4
    k_rb = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]], np.float32) / 4
    f = lambda m, k: cv2.filter2D(rggb * m, -1, k, borderType=cv2.BORDER_REFLECT_101)
    return np.stack([f(r, k_rb), f(g, k_g), f(b, k_rb)], -1)


_G_AT_RB = np.array([[0, 0, -1, 0, 0], [0, 0, 2, 0, 0], [-1, 2, 4, 2, -1],
                     [0, 0, 2, 0, 0], [0, 0, -1, 0, 0]], np.float32) / 8
_RB_AT_G_ROW = np.array([[0, 0, .5, 0, 0], [0, -1, 0, -1, 0], [-1, 4, 5, 4, -1],
                         [0, -1, 0, -1, 0], [0, 0, .5, 0, 0]], np.float32) / 8
_RB_AT_G_COL = np.ascontiguousarray(_RB_AT_G_ROW.T)
_RB_AT_BR = np.array([[0, 0, -1.5, 0, 0], [0, 2, 0, 2, 0], [-1.5, 0, 6, 0, -1.5],
                      [0, 2, 0, 2, 0], [0, 0, -1.5, 0, 0]], np.float32) / 8


def demosaic_malvar(rggb):
    """Malvar-He-Cutler gradient-corrected linear demosaicing."""
    f = lambda k: cv2.filter2D(rggb, -1, k, borderType=cv2.BORDER_REFLECT)
    g_rb, rb_row, rb_col, rb_br = f(_G_AT_RB), f(_RB_AT_G_ROW), f(_RB_AT_G_COL), f(_RB_AT_BR)
    r, g, b = rggb.copy(), rggb.copy(), rggb.copy()
    g[0::2, 0::2] = g_rb[0::2, 0::2]
    g[1::2, 1::2] = g_rb[1::2, 1::2]
    r[0::2, 1::2] = rb_row[0::2, 1::2]
    r[1::2, 0::2] = rb_col[1::2, 0::2]
    r[1::2, 1::2] = rb_br[1::2, 1::2]
    b[1::2, 0::2] = rb_row[1::2, 0::2]
    b[0::2, 1::2] = rb_col[0::2, 1::2]
    b[0::2, 0::2] = rb_br[0::2, 0::2]
    return np.stack([r, g, b], -1)


def packed_rgb(packed):
    return np.stack([packed[..., 0], (packed[..., 1] + packed[..., 2]) / 2, packed[..., 3]], -1)


def wb_gray_world(rgb):
    n = rgb.reshape(-1, 3).mean(0)
    return n / n[1]


def wb_shades_of_gray(rgb, p=6):
    n = (np.clip(rgb, 0, None).reshape(-1, 3) ** p).mean(0) ** (1 / p)
    return n / n[1]


def wb_gray_edge(rgb, p=6):
    s = cv2.GaussianBlur(rgb, (0, 0), 2)
    gx = cv2.Sobel(s, cv2.CV_32F, 1, 0)
    gy = cv2.Sobel(s, cv2.CV_32F, 0, 1)
    n = (np.sqrt(gx ** 2 + gy ** 2).reshape(-1, 3) ** p).mean(0) ** (1 / p)
    return n / n[1]


def cam_to_srgb(color_matrix, neutral):
    """3x3 map from raw camera RGB to linear sRGB: XYZ->sRGB . cam->XYZ . WB.

    The row normalisation (and leaving XYZ->sRGB as it is) matches the renderer behind the
    SIDD ground truth; this was verified by fitting the matrix on a full image.
    """
    xyz_to_cam = np.asarray(color_matrix, np.float64).reshape(3, 3)
    xyz_to_cam = xyz_to_cam / xyz_to_cam.sum(1, keepdims=True)
    cam_to_xyz = np.linalg.inv(xyz_to_cam)
    cam_to_xyz = cam_to_xyz / cam_to_xyz.sum(1, keepdims=True)
    return XYZ_TO_SRGB @ cam_to_xyz @ np.diag(1.0 / np.asarray(neutral, np.float64))


def apply_matrix(rgb, m):
    return (rgb.reshape(-1, 3) @ np.asarray(m, np.float32).T).reshape(rgb.shape)


def srgb_oetf(lin):
    lin = np.clip(lin, 0, 1)
    return np.where(lin <= 0.0031308, 12.92 * lin,
                    1.055 * np.maximum(lin, 1e-8) ** (1 / 2.4) - 0.055).astype(np.float32)


def srgb_inverse_oetf(s):
    return np.where(s <= 0.04045, s / 12.92, ((s + 0.055) / 1.055) ** 2.4).astype(np.float32)


def rgb_to_ycbcr(rgb):
    y = rgb @ np.array([.299, .587, .114], np.float32)
    return y, (rgb[..., 2] - y) * 0.564, (rgb[..., 0] - y) * 0.713


def ycbcr_to_rgb(y, cb, cr):
    r = y + cr / 0.713
    b = y + cb / 0.564
    return np.stack([r, (y - .299 * r - .114 * b) / .587, b], -1)


def guided_filter(guide, src, radius, eps):
    """He et al. guided filter, single-channel guide, box windows."""
    k = (2 * radius + 1,) * 2
    box = lambda x: cv2.blur(x, k, borderType=cv2.BORDER_REFLECT)
    mg, ms = box(guide), box(src)
    a = (box(guide * src) - mg * ms) / (box(guide * guide) - mg * mg + eps)
    b = ms - a * mg
    return box(a) * guide + box(b)


def denoise_packed(packed, a, b, strength=1.0):
    """Noise-adaptive edge-preserving Bayer denoiser.

    Generalised Anscombe stabilises Poisson-Gaussian noise to sigma ~= 1, then a bilateral
    filter runs with a range sigma tied to that unit noise level.
    """
    off = 3 / 8 + b / a ** 2
    out = np.empty_like(packed)
    for c in range(4):
        v = (2 * np.sqrt(np.maximum(packed[..., c] / a + off, 0))).astype(np.float32)
        v = cv2.bilateralFilter(v, 5, 2.5 * strength, 1.5)
        out[..., c] = ((v / 2) ** 2 - off) * a
    return out


def denoise_chroma(rgb, radius, eps):
    """Chroma noise is the most visible: smooth Cb/Cr guided by luma, leave luma alone."""
    y, cb, cr = rgb_to_ycbcr(rgb)
    return ycbcr_to_rgb(y, guided_filter(y, cb, radius, eps), guided_filter(y, cr, radius, eps))


def tone_map(srgb, strength):
    """Global contrast S-curve blended in by `strength` (0 keeps the input)."""
    if strength <= 0:
        return srgb
    return (1 - strength) * srgb + strength * (3 * srgb ** 2 - 2 * srgb ** 3)


def auto_exposure_gain(lin_rgb, target=0.18, max_gain=8.0):
    """Digital gain lifting median luminance to `target`; never darkens."""
    y = lin_rgb @ np.array([.2126, .7152, .0722], np.float32)
    return float(np.clip(target / max(float(np.median(y)), 1e-4), 1.0, max_gain))


def sharpen(srgb, amount, sigma=1.0, noise_gate=0.01):
    """Luma unsharp mask; the gate skips detail below the noise floor."""
    if amount <= 0:
        return srgb
    y, cb, cr = rgb_to_ycbcr(srgb)
    d = y - cv2.GaussianBlur(y, (0, 0), sigma)
    d = np.where(np.abs(d) > noise_gate, d, 0)
    return np.clip(ycbcr_to_rgb(y + amount * d, cb, cr), 0, 1)


def estimate_noise(packed, block=8):
    """Poisson-Gaussian fit var = a * I + b from one noisy packed image.

    The difference of the two green planes cancels most structure. Its block variance is
    compared with block mean; a low quantile per intensity bin rejects remaining texture.
    """
    g = packed[..., 1:3].mean(-1)
    d = (packed[..., 1] - packed[..., 2]) / np.sqrt(2)
    h, w = g.shape[0] // block * block, g.shape[1] // block * block
    blocks = lambda x: x[:h, :w].reshape(h // block, block, w // block, block)
    bm = blocks(g).mean((1, 3)).ravel()
    bv = blocks(d).var((1, 3)).ravel()
    edges = np.quantile(bm, np.linspace(0, 1, 17))
    xs, ys = [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (bm >= lo) & (bm <= hi)
        if m.sum() > 8:
            xs.append(bm[m].mean())
            ys.append(np.quantile(bv[m], 0.3) * 1.25)
    if len(xs) < 3:
        return 1e-7, max(float(np.median(bv)), 1e-9)
    a, b = np.polyfit(np.maximum(xs, 0), ys, 1)
    return float(max(a, 1e-7)), float(max(b, 1e-9))
