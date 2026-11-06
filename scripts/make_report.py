"""Build eval_P2_<TeamID>.md and the app's results table from results/*.json.

  python scripts/make_report.py --team T01 --usns 1XX22CS001,1XX22CS002
"""
import argparse
import json
from pathlib import Path

R = Path("results")
load = lambda n: json.loads((R / f"{n}.json").read_text())

ORDER = ["Classical ISP, gray-world WB", "Classical ISP, corrected AWB", "Learned ISP, end-to-end (4.0M)",
         "Hybrid teacher FP32", "Hybrid student FP32", "Hybrid student FP16", "Hybrid student INT8",
         "Hybrid light student INT8"]
ORACLE = ["Classical ISP, corrected AWB (oracle WB)", "Hybrid teacher FP32 (oracle WB)", "Hybrid student INT8 (oracle WB)"]


def quality_rows(main, names):
    return [[n, f"{main[n]['psnr']:.2f}", f"{main[n]['ssim']:.4f}", f"{main[n]['delta_e']:.2f}", f"{main[n]['lpips']:.4f}"]
            for n in names if n in main]


def table(headers, rows):
    out = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    return "\n".join(out + ["| " + " | ".join(map(str, r)) + " |" for r in rows])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--team", default="TXX")
    ap.add_argument("--usns", default="USN1,USN2,USN3")
    args = ap.parse_args()
    main_r, eff, abl, rob, fail = load("main"), load("efficiency"), load("ablation"), load("robustness"), load("failures")
    q_head = ["Pipeline", "PSNR (dB) ↑", "SSIM ↑", "ΔE2000 ↓", "LPIPS ↓"]

    eff_rows = [[n, f"{v.get('params', 0):,}", f"{v.get('size_mib', 0):.2f}", f"{v.get('gmacs_per_mp', 0):.1f}",
                 f"{v['ms_per_tile']:.0f}", f"{v['sec_per_12mp']:.1f}",
                 f"{v['peak_mb']:.0f}" if "peak_mb" in v else "n/a"]
                for n, v in eff.items() if isinstance(v, dict)]
    rob_rows = []
    for cond, per in rob.items():
        for name, m in per.items():
            rob_rows.append([cond, name, f"{m['psnr']:.2f}", f"{m['ssim']:.4f}", f"{m['delta_e']:.2f}", f"{m['lpips']:.4f}"])
    abl_rows = [[n, f"{m['psnr']:.2f}", f"{m['ssim']:.4f}", f"{m['delta_e']:.2f}", f"{m['lpips']:.4f}"] for n, m in abl.items()]
    awb_e = main_r["awb_angular_error_deg"]

    doc = f"""{args.usns}

# Evaluation: P2, Intelligent and Computationally Efficient Smartphone ISP (Team {args.team})

Test set: all 928 blocks (29 instances x 32) of the SIDD-CC benchmark split, not used for training,
tuning or calibration. Input is noisy RAW; target is the clean sRGB ground truth. The blocks ship
without metadata, so white balance and noise are estimated from each scene's blocks ("blind").
All settings were chosen on a held-out validation scene. Test blocks were seen while debugging; see docs/DESIGN.md section 10.

## 1. PSNR, SSIM, Delta E and LPIPS

{table(q_head, quality_rows(main_r, ORDER))}

Diagnostic rows with the true white balance (recovered from ground truth, not available in practice):

{table(q_head, quality_rows(main_r, ORACLE))}

White-balance angular error on the test scenes: gray-world {awb_e['gw']:.2f} deg, corrected AWB {awb_e['ridge']:.2f} deg.

## 2. Model parameters, size, latency and memory

CPU, ONNX Runtime, {eff['_threads']} threads. A tile is 512 x 512 raw pixels (0.26 MP). The 12 MP figure is scaled from the tile.
Peak memory is the working set of a fresh process running one tile, including the runtime.

{table(["Model", "Parameters", "Size (MiB)", "GMAC/MP", "ms/tile", "s per 12 MP", "Peak MB"], eff_rows)}

## 3. Robustness (every 4th block)

{table(["Condition", "Pipeline", "PSNR", "SSIM", "ΔE", "LPIPS"], rob_rows)}

## 4. Ablation study (every 2nd block)

Network variants are the small student, trained for the same 12,000 iterations with one component off.

{table(q_head[:1] + q_head[1:], abl_rows)}

## 5. Failure cases

See docs/FAILURES.md for analysis. Worst blocks of the deployed model:

{table(["Scene", "Camera", "Block", "PSNR", "ΔE"], [[w['scene'], w['cam'], w['block'], f"{w['psnr']:.2f}", f"{w['delta_e']:.2f}"] for w in fail['worst']])}
"""
    Path(f"eval_P2_{args.team}.md").write_text(doc, encoding="utf8")
    summary = dict(headers=q_head, rows=quality_rows(main_r, ORDER))
    (R / "summary.json").write_text(json.dumps(summary, indent=1), encoding="utf8")
    print("wrote", f"eval_P2_{args.team}.md")


if __name__ == "__main__":
    main()
