USN1,USN2,USN3

# Evaluation: P2, Intelligent and Computationally Efficient Smartphone ISP (Team TXX)

Test set: all 928 blocks (29 instances x 32) of the SIDD-CC benchmark split, not used for training,
tuning or calibration. Input is noisy RAW; target is the clean sRGB ground truth. The blocks ship
without metadata, so white balance and noise are estimated from each scene's blocks ("blind").
All settings were chosen on a held-out validation scene. Test blocks were seen while debugging; see docs/DESIGN.md section 10.

## 1. PSNR, SSIM, Delta E and LPIPS

| Pipeline | PSNR (dB) ↑ | SSIM ↑ | ΔE2000 ↓ | LPIPS ↓ |
|---|---|---|---|---|
| Classical ISP, gray-world WB | 30.43 | 0.8643 | 6.29 | 0.2864 |
| Classical ISP, corrected AWB | 33.38 | 0.8668 | 3.94 | 0.2666 |
| Learned ISP, end-to-end (4.0M) | 35.03 | 0.9059 | 3.66 | 0.1749 |
| Hybrid teacher FP32 | 35.70 | 0.9220 | 3.42 | 0.1643 |
| Hybrid student FP32 | 35.48 | 0.9200 | 3.48 | 0.1667 |
| Hybrid student FP16 | 35.48 | 0.9200 | 3.48 | 0.1668 |
| Hybrid student INT8 | 35.23 | 0.9126 | 3.61 | 0.1286 |
| Hybrid light student INT8 | 34.46 | 0.8858 | 3.92 | 0.1582 |

Diagnostic rows with the true white balance (recovered from ground truth, not available in practice):

| Pipeline | PSNR (dB) ↑ | SSIM ↑ | ΔE2000 ↓ | LPIPS ↓ |
|---|---|---|---|---|
| Classical ISP, corrected AWB (oracle WB) | 35.87 | 0.8694 | 2.90 | 0.2567 |
| Hybrid teacher FP32 (oracle WB) | 39.60 | 0.9262 | 2.23 | 0.1542 |
| Hybrid student INT8 (oracle WB) | 38.61 | 0.9162 | 2.51 | 0.1181 |

White-balance angular error on the test scenes: gray-world 5.46 deg, corrected AWB 2.61 deg.

## 2. Model parameters, size, latency and memory

CPU, ONNX Runtime, 2 threads. A tile is 512 x 512 raw pixels (0.26 MP). The 12 MP figure is scaled from the tile.
Peak memory is the working set of a fresh process running one tile, including the runtime.

| Model | Parameters | Size (MiB) | GMAC/MP | ms/tile | s per 12 MP | Peak MB |
|---|---|---|---|---|---|---|
| Hybrid teacher FP32 | 4,022,444 | 15.53 | 36.3 | 218 | 10.0 | 240 |
| Hybrid student FP32 | 947,036 | 3.62 | 11.8 | 33 | 1.5 | 95 |
| Hybrid student FP16 | 947,036 | 1.82 | 11.8 | 32 | 1.5 | 96 |
| Hybrid student INT8 | 947,036 | 0.98 | 11.8 | 15 | 0.7 | 91 |
| Light student INT8 | 145,396 | 0.19 | 2.5 | 11 | 0.5 | 86 |
| Classical ISP | 0 | 0.00 | 0.0 | 39 | 1.8 | n/a |

## 3. Robustness (every 4th block)

| Condition | Pipeline | PSNR | SSIM | ΔE | LPIPS |
|---|---|---|---|---|---|
| clean input (as captured) | Classical ISP, corrected AWB | 33.33 | 0.8637 | 3.97 | 0.2681 |
| clean input (as captured) | Learned ISP, end-to-end (4.0M) | 34.95 | 0.9041 | 3.73 | 0.1784 |
| clean input (as captured) | Hybrid student INT8 | 35.21 | 0.9101 | 3.63 | 0.1326 |
| extra noise x2 | Classical ISP, corrected AWB | 31.65 | 0.8064 | 4.46 | 0.3726 |
| extra noise x2 | Learned ISP, end-to-end (4.0M) | 34.21 | 0.8899 | 3.99 | 0.1919 |
| extra noise x2 | Hybrid student INT8 | 34.43 | 0.8921 | 3.91 | 0.1550 |
| extra noise x4 | Classical ISP, corrected AWB | 29.48 | 0.7184 | 5.26 | 0.5037 |
| extra noise x4 | Learned ISP, end-to-end (4.0M) | 32.71 | 0.8705 | 4.70 | 0.2162 |
| extra noise x4 | Hybrid student INT8 | 33.28 | 0.8579 | 4.46 | 0.1925 |
| underexposed -1 EV | Classical ISP, corrected AWB | 30.99 | 0.7743 | 4.67 | 0.4183 |
| underexposed -1 EV | Learned ISP, end-to-end (4.0M) | 33.70 | 0.8817 | 4.22 | 0.1963 |
| underexposed -1 EV | Hybrid student INT8 | 33.75 | 0.8687 | 4.29 | 0.1754 |
| underexposed -2 EV | Classical ISP, corrected AWB | 28.06 | 0.6393 | 5.94 | 0.5791 |
| underexposed -2 EV | Learned ISP, end-to-end (4.0M) | 31.34 | 0.8388 | 5.49 | 0.2423 |
| underexposed -2 EV | Hybrid student INT8 | 31.46 | 0.7982 | 5.90 | 0.2493 |
| WB error +15% | Classical ISP, corrected AWB | 30.57 | 0.8672 | 5.67 | 0.2747 |
| WB error +15% | Learned ISP, end-to-end (4.0M) | 34.95 | 0.9041 | 3.73 | 0.1784 |
| WB error +15% | Hybrid student INT8 | 31.45 | 0.9069 | 5.49 | 0.1572 |
| hot pixels 0.1% | Classical ISP, corrected AWB | 29.28 | 0.8147 | 4.78 | 0.5745 |
| hot pixels 0.1% | Learned ISP, end-to-end (4.0M) | 32.59 | 0.8808 | 4.36 | 0.2859 |
| hot pixels 0.1% | Hybrid student INT8 | 34.10 | 0.8972 | 3.80 | 0.2303 |

## 4. Ablation study (every 2nd block)

Network variants are the small student, trained for the same 12,000 iterations with one component off.

| Pipeline | PSNR (dB) ↑ | SSIM ↑ | ΔE2000 ↓ | LPIPS ↓ |
|---|---|---|---|---|
| Baseline: small student, distilled, augmented | 34.95 | 0.9037 | 3.67 | 0.2073 |
| - learned AWB correction (gray-world) | 31.26 | 0.8979 | 5.93 | 0.2383 |
| + noise conditioning planes | 34.81 | 0.9004 | 3.69 | 0.2111 |
| + noise planes, - augmentation | 34.79 | 0.9003 | 3.71 | 0.2156 |
| + noise planes, - distillation | 34.75 | 0.8968 | 3.69 | 0.2167 |
| - all learned restoration (classical only) | 33.36 | 0.8657 | 3.98 | 0.2674 |
| Classical - bilateral raw denoise | 30.37 | 0.6902 | 4.37 | 0.5400 |
| Classical - chroma denoise | 30.61 | 0.7534 | 6.44 | 0.3273 |
| Classical - both denoisers | 25.60 | 0.4361 | 10.74 | 0.5759 |

## 5. Failure cases

See docs/FAILURES.md for analysis. Worst blocks of the deployed model:

| Scene | Camera | Block | PSNR | ΔE |
|---|---|---|---|---|
| 0020 | GP | 21 | 24.60 | 4.24 |
| 0020 | GP | 24 | 24.65 | 10.44 |
| 0020 | GP | 25 | 24.83 | 10.15 |
| 0020 | GP | 23 | 25.57 | 6.35 |
| 0106 | GP | 20 | 25.68 | 2.63 |
| 0020 | GP | 31 | 25.75 | 9.28 |
