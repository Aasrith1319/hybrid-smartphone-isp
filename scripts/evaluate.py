"""Benchmark every pipeline on the 928 unseen SIDD-CC test blocks.

  python scripts/evaluate.py main        comparison table          -> results/main.json
  python scripts/evaluate.py ablation    component ablations       -> results/ablation.json
  python scripts/evaluate.py robustness  degraded inputs           -> results/robustness.json
  python scripts/evaluate.py failures    worst blocks + breakdowns -> results/failures.json

Test blocks ship without metadata, so white balance and noise are estimated ("blind").
Rows tagged "oracle WB" use the neutral recovered from ground truth and only show how much
error comes from colour estimation rather than restoration.
"""
import json
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from isp import awb, classical as C, metrics as Me, models as M, pipeline as P, testset as T

DEV = "cuda"


def torch_core(ckpt, arch, in_ch=6):
    net = M.build(arch, in_ch=in_ch).to(DEV).eval()
    net.load_state_dict(torch.load(ckpt, map_location=DEV, weights_only=True))

    @torch.no_grad()
    def core(x):
        return net(torch.from_numpy(x).to(DEV)).cpu().numpy()
    return core


def onnx_core(path):
    s = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    return lambda x: s.run(None, {"x": x})[0]


def restorer(kind, **kw):
    return P.Restorer(kind, tile=512, **kw)


class Ctx:
    """Per-scene estimates, computed once."""

    def __init__(self, stride=1):
        self.scenes = T.load()
        self.ridge = awb.RidgeAWB.load("weights/awb.json")
        self.stride = stride
        self.est = {}
        for s in self.scenes:
            cam = T.CAMS[s.cam]
            pk = T.pooled_packed(s.noisy, cam["cfa"])
            rgb = T.half_rgb(pk)
            a, b = C.estimate_noise(pk)
            self.est[s.sid] = dict(a=a, b=b, gw=C.wb_gray_world(rgb),
                                   ridge=self.ridge.predict(awb.features(rgb, s.cam)),
                                   oracle=T.oracle_neutral(s))

    def scene(self, s, wb):
        cam, e = T.CAMS[s.cam], self.est[s.sid]
        return P.Scene(cam["cfa"], cam["color_matrix"], e[wb], e["a"], e["b"])


def run(ctx, fn, wb="ridge", transform=None, per_block=False):
    """fn(block, Scene) -> sRGB. Returns mean metrics over all selected blocks."""
    preds, gts, tags = [], [], []
    for s in ctx.scenes:
        scene = ctx.scene(s, wb)
        for i in range(0, 32, ctx.stride):
            block = s.noisy[i]
            sc = scene
            if transform:
                block, sc = transform(block, scene, np.random.RandomState(i))
            preds.append(fn(block, sc).astype(np.float32))
            gts.append(s.gt[i])
            tags.append((s.sid, i))
    m = Me.all_metrics(preds, gts, DEV)
    return (m, preds, gts, tags) if per_block else m


def pipelines():
    """name -> (function, white-balance source)."""
    out = {}
    out["Classical ISP, gray-world WB"] = (lambda b, s: P.classical_isp(b, s), "gw")
    out["Classical ISP, corrected AWB"] = (lambda b, s: P.classical_isp(b, s), "ridge")
    if Path("weights/e2e.pt").exists():
        e2e = restorer(torch_core("weights/e2e.pt", "teacher", in_ch=4), noise_input=False)
        out["Learned ISP, end-to-end (4.0M)"] = (lambda b, s: P.learned_isp(b, s, e2e), "ridge")
    for name, core, cond in [("Hybrid teacher FP32", torch_core("weights/teacher.pt", "teacher"), True),
                             ("Hybrid student FP32", onnx_core("weights/student_fp32.onnx"), False),
                             ("Hybrid student FP16", onnx_core("weights/student_fp16.onnx"), False),
                             ("Hybrid student INT8", onnx_core("weights/student_int8.onnx"), False),
                             ("Hybrid light student INT8", onnx_core("weights/student_s_int8.onnx"), False)]:
        r = restorer(core, conditioned=cond)
        out[name] = ((lambda rr: lambda b, s: P.hybrid_isp(b, s, rr))(r), "ridge")
    return out


def cmd_main():
    ctx = Ctx()
    res = {}
    p = pipelines()
    for name, (fn, wb) in p.items():
        res[name] = run(ctx, fn, wb)
        print(name, res[name], flush=True)
        Path("results/main.json").write_text(json.dumps(res, indent=1))
    for name in ("Classical ISP, corrected AWB", "Hybrid teacher FP32", "Hybrid student INT8"):
        res[name + " (oracle WB)"] = run(ctx, p[name][0], "oracle")
        print(name, "oracle", res[name + " (oracle WB)"], flush=True)
    res["awb_angular_error_deg"] = {
        k: float(np.mean([awb.angular_error(ctx.est[s.sid][k], ctx.est[s.sid]["oracle"]) for s in ctx.scenes]))
        for k in ("gw", "ridge")}
    Path("results/main.json").write_text(json.dumps(res, indent=1))


def cmd_e2e():
    """Recompute only the end-to-end rows in the saved tables."""
    name = "Learned ISP, end-to-end (4.0M)"
    fn = pipelines()[name][0]
    main = json.loads(Path("results/main.json").read_text())
    main[name] = run(Ctx(), fn)
    Path("results/main.json").write_text(json.dumps(main, indent=1))
    print(main[name], flush=True)
    rob = json.loads(Path("results/robustness.json").read_text())
    ctx = Ctx(stride=4)
    for cond, tr in conditions().items():
        rob[cond][name] = run(ctx, fn, "ridge", tr)
        print(cond, rob[cond][name], flush=True)
    Path("results/robustness.json").write_text(json.dumps(rob, indent=1))


def cmd_ablation():
    """Each variant is the small student, trained for the same budget, one component off."""
    ctx = Ctx(stride=2)
    res = {}
    hyb = lambda ckpt, cond=False: (lambda r: lambda b, s: P.hybrid_isp(b, s, r))(
        restorer(torch_core(ckpt, "student_s"), conditioned=cond))
    rows = [("Baseline: small student, distilled, augmented", hyb("weights/abl_nonoise.pt"), "ridge"),
            ("- learned AWB correction (gray-world)", hyb("weights/abl_nonoise.pt"), "gw"),
            ("+ noise conditioning planes", hyb("weights/abl_full.pt", True), "ridge"),
            ("+ noise planes, - augmentation", hyb("weights/abl_noaug.pt", True), "ridge"),
            ("+ noise planes, - distillation", hyb("weights/abl_nokd.pt", True), "ridge"),
            ("- all learned restoration (classical only)", lambda b, s: P.classical_isp(b, s), "ridge"),
            ("Classical - bilateral raw denoise", lambda b, s: P.classical_isp(b, s, P.ClassicalParams(denoise=False)), "ridge"),
            ("Classical - chroma denoise", lambda b, s: P.classical_isp(b, s, P.ClassicalParams(chroma=False)), "ridge"),
            ("Classical - both denoisers", lambda b, s: P.classical_isp(b, s, P.ClassicalParams(False, False)), "ridge")]
    for name, fn, wb in rows:
        res[name] = run(ctx, fn, wb)
        print(name, res[name], flush=True)
        Path("results/ablation.json").write_text(json.dumps(res, indent=1))


def conditions():
    def noise(k):
        def f(b, s, rng):
            v = k * (s.a * np.clip(b, 0, 1) + s.b)
            return (b + rng.randn(*b.shape).astype(np.float32) * np.sqrt(v).astype(np.float32),
                    P.Scene(s.cfa, s.color_matrix, s.neutral, s.a * (1 + k), s.b * (1 + k)))
        return f

    def underexpose(ev):
        def f(b, s, rng):
            # A shorter exposure, then the camera's digital gain back up: noise rises 2^ev in shot
            # variance and 4^ev in read variance, while brightness stays the same.
            g = 2.0 ** ev
            extra = (g - 1) * s.a * np.clip(b, 0, 1) + (g * g - 1) * max(s.b, 2e-6)
            return (b + rng.randn(*b.shape).astype(np.float32) * np.sqrt(extra).astype(np.float32),
                    P.Scene(s.cfa, s.color_matrix, s.neutral, s.a * g, max(s.b, 2e-6) * g * g))
        return f

    def wb_err(deg):
        def f(b, s, rng):
            n = s.neutral * np.array([np.exp(deg), 1, np.exp(-deg)])
            return b, P.Scene(s.cfa, s.color_matrix, n / n[1], s.a, s.b)
        return f

    def hot(p):
        def f(b, s, rng):
            b = b.copy()
            b[rng.rand(*b.shape) < p] = 1.0
            return b, s
        return f

    return {"clean input (as captured)": None, "extra noise x2": noise(1), "extra noise x4": noise(3),
            "underexposed -1 EV": underexpose(1), "underexposed -2 EV": underexpose(2),
            "WB error +15%": wb_err(.15), "hot pixels 0.1%": hot(.001)}


def cmd_robustness():
    ctx = Ctx(stride=4)
    p = pipelines()
    res = {}
    for cond, tr in conditions().items():
        res[cond] = {}
        for name in ("Classical ISP, corrected AWB", "Learned ISP, end-to-end (4.0M)", "Hybrid student INT8"):
            if name in p:
                res[cond][name] = run(ctx, p[name][0], "ridge", tr)
                print(cond, name, res[cond][name], flush=True)
        Path("results/robustness.json").write_text(json.dumps(res, indent=1))


def cmd_failures():
    ctx = Ctx()
    fn = pipelines()["Hybrid student INT8"][0]
    m, preds, gts, tags = run(ctx, fn, "ridge", per_block=True)
    ps = np.array([Me.psnr(a, b) for a, b in zip(preds, gts)])
    de = np.array([Me.delta_e(a, b) for a, b in zip(preds, gts)])
    by_scene = {}
    for (sid, _), p_, d in zip(tags, ps, de):
        by_scene.setdefault(sid, []).append((p_, d))
    cams = {s.sid: s.cam for s in ctx.scenes}
    rows = [dict(scene=k, cam=cams[k], psnr=float(np.mean([x[0] for x in v])), delta_e=float(np.mean([x[1] for x in v])),
                 awb_err_deg=awb.angular_error(ctx.est[k]["ridge"], ctx.est[k]["oracle"]),
                 a=ctx.est[k]["a"], b=ctx.est[k]["b"]) for k, v in by_scene.items()]
    Path("results/failures").mkdir(exist_ok=True, parents=True)
    worst = []
    noisy_of = {s.sid: s for s in ctx.scenes}
    for i in np.argsort(ps)[:6]:
        sid, b = tags[i]
        s = noisy_of[sid]
        naive = P.classical_isp(s.noisy[b], ctx.scene(s, "ridge"), P.ClassicalParams(False, False))
        panel = np.concatenate([naive, preds[i], gts[i]], 1)
        Image.fromarray((np.clip(panel, 0, 1) * 255).astype(np.uint8)).save(f"results/failures/worst_{sid}_{b}.png")
        worst.append(dict(scene=sid, cam=s.cam, block=int(b), psnr=float(ps[i]), delta_e=float(de[i])))
    Path("results/failures.json").write_text(json.dumps(dict(overall=m, scenes=rows, worst=worst), indent=1))
    print(json.dumps(worst, indent=0))


if __name__ == "__main__":
    Path("results").mkdir(exist_ok=True)
    globals()["cmd_" + sys.argv[1]]()
