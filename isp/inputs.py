"""Turn whatever the user uploads into a linear Bayer frame plus camera hints."""
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image, UnidentifiedImageError

from . import classical as C

CAMS = json.loads((Path(__file__).parent / "cameras.json").read_text())
RAW_EXT = {".dng", ".cr2", ".cr3", ".nef", ".arw", ".orf", ".rw2", ".raf", ".pef", ".srw"}
MAX_PIXELS = 4_000_000


class InputError(ValueError):
    """Raised with a message that is safe and useful to show to the user."""


@dataclass
class Frame:
    bayer: np.ndarray
    cfa: str
    color_matrix: np.ndarray
    source: str
    neutral: Optional[np.ndarray] = None   # as-shot, when the file has it
    camera: Optional[str] = None           # key into cameras.json, enables the AWB bias term
    reference: Optional[np.ndarray] = None  # sRGB target, when one exists
    noise: Optional[tuple] = None          # (a, b) measured on the full frame, for crops
    note: str = ""


def _pattern_string(raw):
    desc = raw.color_desc.decode()
    if len(desc) < 3 or raw.raw_pattern is None or raw.raw_pattern.shape != (2, 2):
        raise InputError("Only 2x2 Bayer sensors are supported. X-Trans and Foveon files are not.")
    cfa = "".join(desc[i].lower() for i in raw.raw_pattern.ravel())
    if cfa not in ("rggb", "grbg", "gbrg", "bggr"):
        raise InputError(f"Unsupported colour filter pattern: {cfa}.")
    return cfa


def load_raw_file(path):
    import rawpy
    try:
        raw = rawpy.imread(str(path))
    except Exception as e:
        raise InputError(f"Could not decode this RAW file: {e}") from e
    with raw:
        cfa = _pattern_string(raw)
        img = raw.raw_image_visible.astype(np.float32)
        black = np.array(raw.black_level_per_channel, np.float32)
        pattern = raw.raw_pattern
        sub = np.empty_like(img)
        for dy in range(2):
            for dx in range(2):
                sub[dy::2, dx::2] = black[pattern[dy, dx]]
        white = float(raw.white_level)
        bayer = (img - sub) / (white - float(black.mean()))
        mul = np.array(raw.camera_whitebalance[:3], np.float64)
        neutral = (mul[1] / mul) if mul.min() > 0 else None
        cm = np.array(raw.rgb_xyz_matrix[:3], np.float64)
    if np.abs(cm).sum() < 1e-6:
        cm = np.array(CAMS["IP"]["color_matrix"]).reshape(3, 3)
    return Frame(bayer.astype(np.float32), cfa, cm, "RAW file", neutral=neutral)


def load_bayer_array(path):
    """.npz samples carry metadata; bare .npy needs to already be normalised Bayer."""
    p = Path(path)
    try:
        d = np.load(p, allow_pickle=False)
    except ValueError as e:
        raise InputError(f"Could not read this array file: {e}") from e
    if p.suffix == ".npz":
        cam = str(d["camera"])
        return Frame(d["bayer"].astype(np.float32), str(d["cfa"]), d["color_matrix"], "SIDD-CC sample",
                     neutral=d["neutral"], camera=cam, reference=d["gt"].astype(np.float32) / 255,
                     noise=(float(d["a"]), float(d["b"])))
    if d.ndim != 2:
        raise InputError("A .npy file must hold one 2-D Bayer mosaic.")
    a = d.astype(np.float32)
    if a.max() > 1.5:
        a = a / (2 ** int(np.ceil(np.log2(a.max() + 1))) - 1)
    return Frame(a, "rggb", np.array(CAMS["IP"]["color_matrix"]).reshape(3, 3), "Bayer array",
                 note="No metadata: assumed RGGB and a generic colour matrix.")


SIMULATED_NEUTRAL = (0.55, 1.0, 0.55)


def unprocess(srgb, neutral=SIMULATED_NEUTRAL, camera="IP"):
    """Simulate the sensor image behind an sRGB picture: invert gamma, colour and white
    balance, then mosaic. It lets ordinary JPEGs exercise the RAW pipeline, with the
    original picture as reference."""
    cm = np.array(CAMS[camera]["color_matrix"]).reshape(3, 3)
    ccm = C.cam_to_srgb(cm, [1, 1, 1])
    cam_wb = C.apply_matrix(C.srgb_inverse_oetf(srgb), np.linalg.inv(ccm))
    cam = np.clip(cam_wb, 0, 1) * np.asarray(neutral, np.float32)
    h, w = cam.shape[0] // 2 * 2, cam.shape[1] // 2 * 2
    bayer = np.empty((h, w), np.float32)
    bayer[0::2, 0::2], bayer[0::2, 1::2] = cam[0:h:2, 0:w:2, 0], cam[0:h:2, 1:w:2, 1]
    bayer[1::2, 0::2], bayer[1::2, 1::2] = cam[1:h:2, 0:w:2, 1], cam[1:h:2, 1:w:2, 2]
    return bayer, cm


def load_picture(path):
    try:
        img = Image.open(path).convert("RGB")
    except (UnidentifiedImageError, OSError) as e:
        raise InputError("This file is not a readable image or RAW file.") from e
    img.thumbnail((2000, 2000))
    srgb = np.asarray(img, np.float32) / 255
    bayer, cm = unprocess(srgb)
    h, w = bayer.shape
    return Frame(bayer, "rggb", cm, "sRGB picture, unprocessed to simulated RAW", reference=srgb[:h, :w], camera="IP",
                 neutral=np.array(SIMULATED_NEUTRAL),
                 note="This is not real sensor data: the picture was inverted through the ISP. "
                      "Use the noise slider to simulate a sensor; the reference is the uploaded picture.")


def load(path):
    p = Path(path)
    ext = p.suffix.lower()
    if ext in RAW_EXT:
        frame = load_raw_file(p)
    elif ext in (".npy", ".npz"):
        frame = load_bayer_array(p)
    else:
        frame = load_picture(p)
    if frame.bayer.ndim != 2 or min(frame.bayer.shape) < 64:
        raise InputError("The image is too small: at least 64 x 64 pixels are needed.")
    return frame


def bin_frame(frame):
    """Keep a whole big frame inside the latency budget: average NxN Bayer tiles of equal
    colour. Binning also lowers noise, by a factor of N in standard deviation."""
    h, w = frame.bayer.shape
    k = int(np.ceil(np.sqrt(h * w / MAX_PIXELS)))
    if k <= 1:
        return frame, 1
    canvas, _ = C.to_canonical(frame.bayer, frame.cfa)
    packed = C.pack(canvas)
    ph, pw = packed.shape[0] // k * k, packed.shape[1] // k * k
    binned = packed[:ph, :pw].reshape(ph // k, k, pw // k, k, 4).mean((1, 3))
    frame.bayer = np.ascontiguousarray(C.unpack(binned))
    frame.cfa = "rggb"
    if frame.reference is not None:
        frame.reference = None
    frame.note = (frame.note + f" Frame binned {k}x{k} to fit the latency budget.").strip()
    return frame, k
