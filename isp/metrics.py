"""Image-quality metrics: PSNR, SSIM, CIEDE2000 and LPIPS."""
import numpy as np
from skimage.color import deltaE_ciede2000, rgb2lab
from skimage.metrics import structural_similarity

_lpips = None


def psnr(a, b):
    return float(-10 * np.log10(np.mean((a.astype(np.float64) - b) ** 2) + 1e-12))


def ssim(a, b):
    return float(structural_similarity(a, b, channel_axis=-1, data_range=1.0))


def delta_e(a, b):
    return float(deltaE_ciede2000(rgb2lab(a), rgb2lab(b)).mean())


def lpips_batch(a, b, device="cuda"):
    """a, b: (N, H, W, 3) in [0, 1]. Returns one value per image."""
    global _lpips
    import lpips
    import torch
    if _lpips is None:
        _lpips = lpips.LPIPS(net="alex", verbose=False).to(device).eval()
    t = lambda x: torch.from_numpy(np.ascontiguousarray(x.transpose(0, 3, 1, 2))).float().to(device) * 2 - 1
    with torch.no_grad():
        return _lpips(t(a), t(b)).flatten().cpu().numpy()


def all_metrics(preds, gts, device="cuda"):
    """Mean metrics over a list of (H, W, 3) float images."""
    p = np.mean([psnr(a, b) for a, b in zip(preds, gts)])
    s = np.mean([ssim(a, b) for a, b in zip(preds, gts)])
    d = np.mean([delta_e(a, b) for a, b in zip(preds, gts)])
    lp = np.concatenate([lpips_batch(np.stack(preds[i:i + 32]), np.stack(gts[i:i + 32]), device)
                         for i in range(0, len(preds), 32)]).mean()
    return dict(psnr=float(p), ssim=float(s), delta_e=float(d), lpips=float(lp))
