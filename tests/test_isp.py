"""Fast checks of the classical stages and the tiling logic. Run: python -m pytest tests"""
import numpy as np
import pytest

from isp import awb, classical as C, inputs, pipeline as P


def bayer_of(rgb, cfa):
    out = np.empty(rgb.shape[:2], np.float32)
    ch = {"r": 0, "g": 1, "b": 2}
    for i, c in enumerate(cfa):
        y, x = divmod(i, 2)
        out[y::2, x::2] = rgb[y::2, x::2, ch[c]]
    return out


def test_pack_roundtrip():
    x = np.random.rand(16, 24).astype(np.float32)
    assert np.array_equal(C.unpack(C.pack(x)), x)


@pytest.mark.parametrize("cfa", ["rggb", "grbg", "gbrg", "bggr"])
def test_canonical_phase(cfa):
    rgb = np.zeros((32, 32, 3), np.float32)
    rgb[..., 0], rgb[..., 1], rgb[..., 2] = .2, .5, .8
    canvas, (y0, x0) = C.to_canonical(bayer_of(rgb, cfa), cfa)
    assert np.allclose(C.pack(canvas).mean((0, 1)), [.2, .5, .5, .8], atol=1e-4)
    assert np.array_equal(canvas[y0:y0 + 32, x0:x0 + 32], bayer_of(rgb, cfa))


@pytest.mark.parametrize("fn", [C.demosaic_bilinear, C.demosaic_malvar])
def test_demosaic_keeps_flat_colour(fn):
    rgb = np.full((32, 32, 3), [.2, .5, .8], np.float32)
    assert np.allclose(fn(bayer_of(rgb, "rggb"))[4:-4, 4:-4], [.2, .5, .8], atol=1e-5)


def test_oetf_inverse():
    x = np.linspace(0, 1, 101, dtype=np.float32)
    assert np.allclose(C.srgb_inverse_oetf(C.srgb_oetf(x)), x, atol=1e-5)


def test_noise_estimate_recovers_parameters():
    rng = np.random.RandomState(0)
    clean = np.repeat(np.linspace(.02, .9, 512, dtype=np.float32)[:, None], 512, 1)
    a, b = 1e-3, 1e-5
    noisy = clean + rng.randn(512, 512).astype(np.float32) * np.sqrt(a * clean + b)
    ea, _ = C.estimate_noise(C.pack(noisy))
    assert .7 * a < ea < 1.4 * a


def test_angular_error():
    assert awb.angular_error([1, 1, 1], [2, 2, 2]) == pytest.approx(0, abs=1e-5)
    assert awb.angular_error([1, 0, 0], [0, 1, 0]) == pytest.approx(90)


def test_restorer_tiles_match_single_pass():
    """A local core must give identical output however the image is tiled."""
    core = lambda x: np.repeat(x[:, :1], 12, 1)
    x = np.random.rand(100, 140, 4).astype(np.float32)
    assert np.allclose(P.Restorer(core, tile=64, overlap=8)(x, 1e-4, 1e-6),
                       P.Restorer(core, tile=512)(x, 1e-4, 1e-6), atol=1e-6)


def test_bad_inputs_are_rejected(tmp_path):
    p = tmp_path / "bad.jpg"
    p.write_bytes(b"not an image")
    with pytest.raises(inputs.InputError):
        inputs.load(p)
    q = tmp_path / "three_d.npy"
    np.save(q, np.zeros((4, 4, 4)))
    with pytest.raises(inputs.InputError):
        inputs.load(q)
