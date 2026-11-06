"""Runtime engine used by the app: adaptive routing around the three ISP paths."""
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import onnxruntime as ort

from . import awb, classical as C, inputs, pipeline as P

WEIGHTS = Path(__file__).resolve().parents[1] / "weights"
UNDEREXPOSED_MEDIAN = 0.03


@dataclass
class Decision:
    rel_noise: float
    a: float
    b: float
    neutral: np.ndarray
    wb_source: str
    exposure: float


class Engine:
    def __init__(self):
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 0
        load = lambda f: ort.InferenceSession(str(WEIGHTS / f), opts, providers=["CPUExecutionProvider"])
        self.full = load("student_int8.onnx")
        light = WEIGHTS / "student_s_int8.onnx"
        self.light = load(light.name) if light.exists() else self.full
        self.awb = awb.RidgeAWB.load(WEIGHTS / "awb.json")

    def analyse(self, frame, auto_exposure=True):
        canvas, _ = C.to_canonical(frame.bayer, frame.cfa, pad=2)
        packed = C.pack(canvas)
        a, b = frame.noise or C.estimate_noise(packed)
        rgb = C.packed_rgb(packed)
        if frame.neutral is not None:
            neutral, src = np.asarray(frame.neutral, np.float64), "as-shot metadata"
        else:
            import cv2
            small = cv2.GaussianBlur(rgb.astype(np.float32), (0, 0), 1.5)
            neutral, src = self.awb.predict(awb.features(small, frame.camera)), "estimated (corrected AWB)"
        rel = P.relative_noise(packed, a, b)
        exposure = 1.0
        if auto_exposure and float(np.median(packed)) < UNDEREXPOSED_MEDIAN:
            exposure = C.auto_exposure_gain(rgb / np.asarray(neutral, np.float32))
        return Decision(rel, a, b, neutral, src, exposure)

    def run(self, frame, finish_look=True, battery_saver=False):
        d = self.analyse(frame, auto_exposure=finish_look)
        scene = P.Scene(frame.cfa, frame.color_matrix, d.neutral, d.a, d.b)
        look = P.Look.display(d.rel_noise, d.exposure) if finish_look else P.Look()

        t = time.perf_counter()
        classical, c_stages = P.classical_isp(frame.bayer, scene, look=look, want_stages=True)
        t_classical = time.perf_counter() - t

        # The light network costs 1-3 dB at every noise level, so it is a budget choice, not
        # something noise can decide (results/route.json).
        session = self.light if battery_saver else self.full
        restorer = P.Restorer(lambda x: session.run(None, {"x": x})[0], tile=256)
        t = time.perf_counter()
        hybrid, h_stages = P.hybrid_isp(frame.bayer, scene, restorer, look=look, want_stages=True)
        t_hybrid = time.perf_counter() - t

        return dict(decision=d, classical=classical, hybrid=hybrid, stages={**c_stages, **h_stages},
                    t_classical=t_classical, t_hybrid=t_hybrid)


def add_sensor_noise(frame, shot, read, seed=0):
    """Simulate a noisier capture on top of the input."""
    if shot <= 0 and read <= 0:
        return frame
    rng = np.random.RandomState(seed)
    var = shot * np.clip(frame.bayer, 0, 1) + read
    frame.bayer = frame.bayer + rng.randn(*frame.bayer.shape).astype(np.float32) * np.sqrt(var).astype(np.float32)
    if frame.noise:
        frame.noise = (frame.noise[0] + shot, frame.noise[1] + read)
    return frame


__all__ = ["Engine", "add_sensor_noise", "inputs"]
