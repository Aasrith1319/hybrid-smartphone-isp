"""Train a restorer (hybrid: learned restoration inside fixed colour stages) or the
end-to-end learned baseline (network does everything).

  python scripts/train.py --name teacher --arch teacher --iters 60000
  python scripts/train.py --name student --arch student --teacher weights/teacher.pt
  python scripts/train.py --name e2e --arch teacher --mode e2e
"""
import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from isp import classical as C, data as D, models as M


def forward(net, batch, mode, use_noise):
    a, b = batch["a"], batch["b"]
    if not use_noise:
        a, b = a * 0, b * 0
    if mode == "e2e":
        f = M.compress(batch["raw"])
        return None, M.srgb_oetf(M.expand(M.bilinear_demosaic(f) + F.pixel_shuffle(net(f), 2)))
    lin = M.restore(net, batch["packed"], a, b)
    return lin, M.render(lin, batch["ccm"])


@torch.no_grad()
def validate(net, tiles, mode, use_noise, dev):
    net.eval()
    ps = []
    for i in range(0, len(tiles), 16):
        chunk = tiles[i:i + 16]
        gains = [1 / np.array(e["neutral"], np.float32) for e, _, _ in chunk]
        packed = np.stack([(C.pack(r) * g[[0, 1, 1, 2]]).transpose(2, 0, 1) for (_, r, _), g in zip(chunk, gains)])
        batch = dict(packed=torch.from_numpy(packed).to(dev),
                     raw=torch.from_numpy(np.stack([C.pack(r).transpose(2, 0, 1) for _, r, _ in chunk])).to(dev),
                     ccm=torch.from_numpy(np.stack([D.color_correction(e) for e, _, _ in chunk])).to(dev),
                     a=torch.tensor([e["a"] for e, _, _ in chunk], device=dev),
                     b=torch.tensor([e["b"] for e, _, _ in chunk], device=dev))
        gt = torch.from_numpy(np.stack([g.transpose(2, 0, 1) for _, _, g in chunk])).to(dev)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = forward(net, batch, mode, use_noise)[1]
        mse = ((out.float() - gt)[..., 8:-8, 8:-8] ** 2).mean((1, 2, 3))
        ps += (-10 * torch.log10(mse)).tolist()
    net.train()
    return float(np.mean(ps))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--arch", default="teacher")
    ap.add_argument("--mode", default="hybrid", choices=["hybrid", "e2e"])
    ap.add_argument("--iters", type=int, default=60000)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--teacher", help="checkpoint for knowledge distillation")
    ap.add_argument("--kd", type=float, default=1.0)
    ap.add_argument("--no-noise-map", action="store_true")
    ap.add_argument("--no-augment", action="store_true")
    ap.add_argument("--resume")
    ap.add_argument("--val-every", type=int, default=2000)
    args = ap.parse_args()

    torch.backends.cudnn.benchmark = True
    dev = "cuda"
    index = D.load_index()
    train, val = D.split(index)
    print(f"train images {len(train)} val images {len(val)}", flush=True)
    tiles = D.val_tiles(val)

    kw = dict(in_ch=4) if args.mode == "e2e" else {}
    net = M.build(args.arch, **kw).to(dev)
    print("params", sum(p.numel() for p in net.parameters()), flush=True)
    teacher = None
    if args.teacher:
        teacher = M.build("teacher").to(dev).eval().requires_grad_(False)
        teacher.load_state_dict(torch.load(args.teacher, map_location=dev, weights_only=True))
    if args.resume:
        net.load_state_dict(torch.load(args.resume, map_location=dev, weights_only=True))

    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, betas=(.9, .9), weight_decay=0)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda i: min(1, (i + 1) / 500) * (.01 + .99 * .5 * (1 + math.cos(math.pi * i / args.iters))))
    loader = torch.utils.data.DataLoader(
        D.Patches(train, size=args.size, length=args.iters * args.batch, augment=not args.no_augment),
        batch_size=args.batch, num_workers=args.workers, persistent_workers=True, prefetch_factor=4, pin_memory=True)

    use_noise = not args.no_noise_map
    best, t0, run = -1, time.time(), 0.0
    log = []
    net.train()
    for it, batch in enumerate(loader, 1):
        batch = {k: v.to(dev, non_blocking=True) for k, v in batch.items()}
        with torch.autocast("cuda", dtype=torch.bfloat16):
            lin, out = forward(net, batch, args.mode, use_noise)
            loss = F.l1_loss(out.float()[..., 8:-8, 8:-8], batch["gt"][..., 8:-8, 8:-8])
            if teacher is not None:
                with torch.no_grad():
                    t_lin = M.restore(teacher, batch["packed"], batch["a"], batch["b"])
                loss = loss + args.kd * F.l1_loss(M.compress(lin.float()), M.compress(t_lin.float()))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        sched.step()
        run += loss.item()
        if it % 200 == 0:
            print(f"it {it} loss {run / 200:.5f} lr {sched.get_last_lr()[0]:.2e} {it / (time.time() - t0):.2f} it/s", flush=True)
            run = 0.0
        if it % args.val_every == 0 or it == args.iters:
            p = validate(net, tiles, args.mode, use_noise, dev)
            log.append(dict(it=it, val_psnr=p, minutes=(time.time() - t0) / 60))
            print(f"== it {it} val PSNR {p:.3f} best {best:.3f}", flush=True)
            Path("results").mkdir(exist_ok=True)
            Path(f"results/train_{args.name}.json").write_text(json.dumps(log, indent=0))
            torch.save(net.state_dict(), f"weights/{args.name}_last.pt")
            if p > best:
                best = p
                torch.save(net.state_dict(), f"weights/{args.name}.pt")


if __name__ == "__main__":
    main()
