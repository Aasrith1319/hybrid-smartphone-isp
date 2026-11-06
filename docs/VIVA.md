# Viva guide

Numbers: blind test, 928 blocks, deployed INT8 unless stated.

## One-paragraph pitch

We measured that the SIDD ground truth is reproduced at 51 dB by plain colour maths, so colour does not need learning.
We keep colour classical and use a small network only for denoising and demosaicing. A 4M teacher trains a 0.95M student,
quantised to INT8 at under 1 MB. It beats the classical ISP on every quality metric, and matches a 4M end-to-end network
with a quarter of the parameters and about 15× lower latency.

## Likely questions

**What is an ISP?** The stages that turn raw sensor data into a photo: black level, white balance, demosaic, denoise,
colour matrix, gamma, tone mapping, sharpening.

**What is Bayer and demosaicing?** Each pixel records one colour in a 2×2 pattern. Demosaicing fills in the other two.

**Why hybrid and not one network?** Colour is exact maths, and a network would only approximate it. The hybrid can also use
camera metadata: with the true white balance it gets 38.6 dB. The end-to-end network cannot use that information.

**Why the square-root compression?** It spreads dark values, roughly evens out shot noise, and suits INT8.

**Why predict a residual?** The network learns only the correction over bilinear demosaic, which is easier, and training
starts from a sensible output.

**What is knowledge distillation?** The student learns from ground truth and from the teacher's output together.

**What does INT8 quantisation do?** Stores weights and activations in 8 bits: 3.6 to 0.98 MiB, about 2× faster. Calibration
on real tiles sets the ranges. Clipping at the 99.9th percentile avoids outliers wasting the range.

**Why did you keep two layers in float?** The first and last convolutions are most sensitive, and they are cheap to keep.

**Why is FP16 not faster?** The CPU has no native FP16 kernels for this model, so it only halves file size.

**What do the metrics mean?** PSNR: pixel error in dB, higher is better. SSIM: structure, up to 1. ΔE: colour difference,
lower is better, about 2 is barely visible. LPIPS: deep-feature perceptual distance, lower is better.

**How do you estimate white balance?** Classical statistics, corrected by ridge regression: 2.6° error on test against 5.5°
for gray-world. DNG files use as-shot metadata instead.

**How do you estimate noise?** The two green planes see nearly the same scene, so their difference is mostly noise.
Measuring it against brightness fits `variance = a·I + b`.

**What is adaptive in your pipeline?** Denoise radius, sharpening and exposure follow measured noise and brightness.
We tried two more ideas, noise inputs and noise-based routing, and rejected both because validation data did not support them.

**How did you avoid overfitting?** Scene 008 was held out for all choices, and the test set was reserved for reporting.
We state in the design doc that debugging touched the test set.

**What is your main weakness?** White balance. Blind PSNR is 3.4 dB below PSNR with the true value.

**Show a failure.** Scene 0020: the AWB error is 5.4°, so a saturated red looks dull and white gets a blue cast. The
texture is clean, so the error comes from colour estimation.

**Would this run on a phone?** ONNX INT8 ports to mobile runtimes, but we measured on a laptop CPU with two threads, not
a phone. That is a limitation.

**What would you do next?** A better illuminant estimator, multi-frame input for extreme noise, and on-device measurement.

## Splitting parts across members

| Part | Files |
|---|---|
| Classical ISP | `isp/classical.py`, `isp/pipeline.py` |
| White balance, noise | `isp/awb.py`, `scripts/fit_awb.py` |
| Models and training | `isp/models.py`, `isp/data.py`, `scripts/train.py` |
| Optimisation | `scripts/export_onnx.py`, `scripts/sweep_quant.py`, `scripts/benchmark.py` |
| Evaluation, app | `scripts/evaluate.py`, `app.py`, `isp/engine.py`, `isp/inputs.py` |
