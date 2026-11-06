"""Download SIDD-CC from the Hugging Face Hub over plain HTTP, resumably.

Training only needs noisy RAW + clean sRGB, so clean RAW is skipped except for one image
used to verify the colour pipeline. Order: test blocks, metadata, then pair 1 of every
instance, then pair 2, so a partial download already covers every camera and scene.
"""
import argparse
import concurrent.futures as cf
import os
from pathlib import Path

import requests
from huggingface_hub import HfApi

REPO = "MhdMohammadi/SIDD-CC"
URL = f"https://huggingface.co/datasets/{REPO}/resolve/main/"


def fetch(item, out):
    path, size = item
    dst = Path(out) / path
    if dst.exists() and dst.stat().st_size == size:
        return 0
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(dst.suffix + ".part")
    for _ in range(5):
        pos = tmp.stat().st_size if tmp.exists() else 0
        try:
            with requests.get(URL + path, stream=True, timeout=60,
                              headers={"Range": f"bytes={pos}-"} if pos else {}) as r:
                r.raise_for_status()
                with open(tmp, "ab" if pos else "wb") as f:
                    for chunk in r.iter_content(1 << 20):
                        f.write(chunk)
            if tmp.stat().st_size == size:
                os.replace(tmp, dst)
                return size
        except requests.RequestException as e:
            print("retry", path, e, flush=True)
    raise RuntimeError(f"failed: {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/SIDD-CC")
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--max-pairs", type=int, default=2, help="RAW pairs per instance")
    args = ap.parse_args()

    tree = [(f.path, f.size) for f in HfApi().list_repo_tree(REPO, repo_type="dataset", recursive=True)
            if hasattr(f, "size") and f.size is not None]
    first = [x for x in tree if "/test/" in x[0] or x[0].endswith((".md", ".txt")) or "METADATA" in x[0]]
    raw = sorted(x for x in tree if "/train/Data" in x[0] and
                 ("NOISY_RAW" in x[0] or "GT_SRGB" in x[0] or "0017_GT_RAW_106" in x[0]))
    idx_of = {}
    for p, _ in raw:
        d, n = os.path.split(p)
        idx_of.setdefault(d.split("/")[-1], set()).add(n.rsplit("_", 1)[1].split(".")[0])
    rank = {d: sorted(v) for d, v in idx_of.items()}
    key = lambda x: rank[x[0].split("/")[-2]].index(x[0].rsplit("_", 1)[1].split(".")[0])
    raw = [x for x in sorted(raw, key=key) if key(x) < args.max_pairs]

    todo = first + raw
    print(len(todo), "files", round(sum(s for _, s in todo) / 2**30, 1), "GiB", flush=True)
    done = 0
    with cf.ThreadPoolExecutor(args.workers) as ex:
        for i, got in enumerate(ex.map(lambda x: fetch(x, args.out), todo)):
            done += got
            if i % 10 == 0:
                print(i, "/", len(todo), round(done / 2**30, 2), "GiB new", flush=True)
    print("complete", flush=True)


if __name__ == "__main__":
    main()
