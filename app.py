"""Gradio front end for the hybrid smartphone ISP."""
import json
from pathlib import Path

import gradio as gr
import numpy as np

from isp import classical as C, inputs, metrics
from isp.engine import Engine, add_sensor_noise

ROOT = Path(__file__).parent
SAMPLES = sorted(str(p) for p in (ROOT / "samples").glob("*.npz"))
engine = Engine()

CSS = """
.gradio-container {max-width: 1180px !important; margin: auto;}
#verdict {font-variant-numeric: tabular-nums;}
"""

INTRO = """
# Hybrid Smartphone ISP
Noisy RAW in, finished sRGB out. A small INT8 network restores the sensor data; exact classical
stages handle white balance, colour and gamma. Start with a sample, or upload a **DNG / camera RAW**,
a **.npy Bayer mosaic**, or a **JPG / PNG** (inverted to simulated RAW first).
"""


def to_u8(x):
    return (np.clip(x, 0, 1) * 255 + .5).astype(np.uint8)


def preview_raw(frame):
    """What the sensor recorded: demosaiced with no colour processing, brightened for viewing."""
    canvas, _ = C.to_canonical(frame.bayer, frame.cfa)
    rgb = C.packed_rgb(C.pack(canvas))
    return to_u8(np.sqrt(np.clip(rgb / max(float(np.quantile(rgb, .99)), 1e-3), 0, 1)))


def score(out, ref):
    """Scored on a 256 px centre crop: full-frame CIEDE2000 alone takes seconds on a shared CPU."""
    h, w = ref.shape[:2]
    y, x = max(h // 2 - 128, 0), max(w // 2 - 128, 0)
    o, r = out[y:y + 256, x:x + 256], ref[y:y + 256, x:x + 256]
    return f"{metrics.psnr(o, r):.2f} dB / SSIM {metrics.ssim(o, r):.3f} / ΔE {metrics.delta_e(o, r):.2f}"


def process(path, sample, noise, look, saver):
    path = path or sample
    if not path:
        raise gr.Error("Pick a sample or upload a file first.")
    try:
        frame = inputs.load(path)
        frame, _ = inputs.bin_frame(frame)
    except inputs.InputError as e:
        raise gr.Error(str(e))
    frame = add_sensor_noise(frame, 4e-4 * noise, 4e-6 * noise ** 2)
    try:
        r = engine.run(frame, finish_look=look, battery_saver=saver)
    except (ValueError, MemoryError, np.linalg.LinAlgError) as e:
        raise gr.Error(f"Processing failed on this input: {e}")

    d = r["decision"]
    lines = [f"**Input:** {frame.source}, {frame.bayer.shape[1]} × {frame.bayer.shape[0]}, CFA {frame.cfa.upper()}",
             f"**Noise estimate:** a = {d.a:.2e}, b = {d.b:.2e}, noise-to-signal {d.rel_noise:.3f}",
             f"**Adaptive control:** {'light' if saver else 'full'} network; exposure gain ×{d.exposure:.2f}",
             f"**White balance:** {d.wb_source}, neutral = {np.round(d.neutral, 3).tolist()}",
             f"**Latency:** hybrid {r['t_hybrid']:.2f} s, classical {r['t_classical']:.2f} s (shared CPU)"]
    if frame.reference is not None and not look and d.exposure == 1.0:
        h, w = frame.reference.shape[:2]
        lines += [f"**Hybrid vs reference (centre crop):** {score(r['hybrid'][:h, :w], frame.reference)}",
                  f"**Classical vs reference (centre crop):** {score(r['classical'][:h, :w], frame.reference)}"]
    elif frame.reference is not None:
        lines.append("Turn off the display look to score against the reference.")
    if frame.note:
        lines.append(f"*{frame.note}*")

    names = {"denoised_raw": "1. Classical denoise (raw)", "white_balanced": "2. White balanced",
             "restored_linear": "3. Learned restoration (linear)", "after_ccm_gamma": "4. Colour matrix + gamma"}
    gallery = [(to_u8(np.sqrt(np.clip(v, 0, 1)) if k != "after_ccm_gamma" else v), n)
               for k, n in names.items() if (v := r["stages"].get(k)) is not None]
    ref = to_u8(frame.reference) if frame.reference is not None else None
    return preview_raw(frame), to_u8(r["classical"]), to_u8(r["hybrid"]), ref, gallery, "\n\n".join(lines)


def build():
    results = ROOT / "results" / "summary.json"
    summary = json.loads(results.read_text()) if results.exists() else None
    with gr.Blocks(title="Hybrid Smartphone ISP") as demo:
        gr.Markdown(INTRO)
        with gr.Row():
            with gr.Column(scale=1, min_width=300):
                upload = gr.File(label="Upload RAW / Bayer / picture", type="filepath")
                sample = gr.Dropdown([(Path(s).stem.replace("_", " "), s) for s in SAMPLES], label="Or pick a sample",
                                     value=next((s for s in SAMPLES if "N6" in s), None))
                noise = gr.Slider(0, 16, 0, step=1, label="Add sensor noise (simulated gain)")
                look = gr.Checkbox(False, label="Display look (auto exposure, tone curve, adaptive sharpening)")
                saver = gr.Checkbox(False, label="Battery saver (light network, lower quality)")
                go = gr.Button("Process", variant="primary")
                report = gr.Markdown(elem_id="verdict")
            with gr.Column(scale=2):
                with gr.Row():
                    raw = gr.Image(label="Sensor data (unprocessed)", format="png")
                    classical = gr.Image(label="Classical ISP", format="png")
                with gr.Row():
                    hybrid = gr.Image(label="Hybrid ISP", format="png")
                    ref = gr.Image(label="Reference, when available", format="png")
        with gr.Accordion("Intermediate stages", open=False):
            stages = gr.Gallery(columns=4, height="auto", show_label=False)
        if summary:
            with gr.Accordion("Benchmark results (928 unseen SIDD-CC blocks)", open=False):
                gr.Dataframe(value=summary["rows"], headers=summary["headers"], interactive=False)
        go.click(process, [upload, sample, noise, look, saver], [raw, classical, hybrid, ref, stages, report])
    return demo


if __name__ == "__main__":
    build().queue(max_size=16).launch(server_name="0.0.0.0", server_port=7860, css=CSS, theme=gr.themes.Soft())
