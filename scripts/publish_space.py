"""Create (if needed) and upload the Hugging Face Space. Run `hf auth login` first.

  python scripts/publish_space.py --repo <username>/hybrid-smartphone-isp

The upload API is used instead of `git push` because Spaces reject plain-git binary files.
"""
import argparse

from huggingface_hub import HfApi

KEEP = ["app.py", "Dockerfile", "requirements.txt", "README.md", "isp/*", "samples/*", "docs/*", "results/summary.json",
        "weights/student_int8.onnx", "weights/student_s_int8.onnx", "weights/awb.json"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    args = ap.parse_args()
    api = HfApi()
    api.create_repo(args.repo, repo_type="space", space_sdk="docker", exist_ok=True)
    api.upload_folder(repo_id=args.repo, repo_type="space", folder_path=".", allow_patterns=KEEP,
                      ignore_patterns=["**/__pycache__/*", "*.pyc"])
    print(f"https://huggingface.co/spaces/{args.repo}")


if __name__ == "__main__":
    main()
