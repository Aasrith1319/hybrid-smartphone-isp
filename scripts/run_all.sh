#!/usr/bin/env bash
# Reproduce everything from a clean checkout. Takes about 6 hours on an RTX 4060 Laptop GPU.
# Set STAGE=after_teacher to skip data preparation and teacher training.
set -euo pipefail
PY=${PY:-python}
STAGE=${STAGE:-all}

if [ "$STAGE" = all ]; then
  $PY scripts/download_data.py
  $PY scripts/prepare_data.py
  $PY scripts/fit_awb.py
  $PY scripts/tune_classical.py
  $PY -u scripts/train.py --name teacher --arch teacher --iters 40000 | tee results/log_teacher.txt
fi

# Deployed student: distilled from the teacher, without noise planes (see docs/DESIGN.md).
$PY -u scripts/train.py --name student --arch student --teacher weights/teacher.pt --no-noise-map --iters 40000 | tee results/log_student.txt

# Ablations share one budget; abl_nonoise doubles as the light route.
ABL="--arch student_s --iters 12000 --val-every 3000"
$PY -u scripts/train.py --name abl_full   $ABL --teacher weights/teacher.pt | tee results/log_abl_full.txt
$PY -u scripts/train.py --name abl_nokd   $ABL | tee results/log_abl_nokd.txt
$PY -u scripts/train.py --name abl_noaug  $ABL --teacher weights/teacher.pt --no-augment | tee results/log_abl_noaug.txt
$PY -u scripts/train.py --name abl_nonoise $ABL --teacher weights/teacher.pt --no-noise-map | tee results/log_abl_nonoise.txt

# End-to-end learned baseline: teacher-sized, no classical stages.
$PY -u scripts/train.py --name e2e --arch teacher --mode e2e --iters 20000 | tee results/log_e2e.txt

$PY scripts/export_onnx.py --ckpt weights/student.pt --arch student --out weights/student
$PY scripts/export_onnx.py --ckpt weights/abl_nonoise.pt --arch student_s --out weights/student_s
