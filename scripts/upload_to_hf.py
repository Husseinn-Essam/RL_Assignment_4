import argparse
import os
import sys
from huggingface_hub import HfApi
from dotenv import load_dotenv
from dotenv import dotenv_values


def find_files(root, exts=None):
    for dirpath, _, filenames in os.walk(root):
        for fn in filenames:
            if exts is None or any(fn.lower().endswith(e) for e in exts):
                yield os.path.join(dirpath, fn)


def main():
    p = argparse.ArgumentParser(description="Upload local model files to Hugging Face Hub")
    p.add_argument("--repo-id", required=True, help="Hugging Face repo id, e.g. username/repo-name")
    p.add_argument("--token", help="HF token (or set env HF_TOKEN)")
    p.add_argument("--source-dir", default="models", help="Directory to upload (default: models)")
    p.add_argument("--create-repo", action="store_true", help="Create the HF repo if it doesn't exist")
    p.add_argument("--ext", action="append", help="File extensions to include, e.g. --ext .pt --ext .pth", default=[".pt", ".pth", ".bin", ".json", ".yaml"])
    p.add_argument("--path-prefix", default="", help="Path prefix inside the HF repo (default: same structure as source-dir)")
    args = p.parse_args()

    path = os.getcwd()
    print(f"Loading .env from {path}")
    load_dotenv(dotenv_path=os.path.join(path, ".env"))
    env_vars = dotenv_values(os.path.join(path, ".env"))
    token = args.token or env_vars.get("HF_TOKEN") or os.environ.get("HF_TOKEN")
    if not token:
        print("ERROR: No HF token provided. Set --token or HF_TOKEN environment variable.")
        sys.exit(1)

    api = HfApi()

    if args.create_repo:
        try:
            api.create_repo(repo_id=args.repo_id, token=token, exist_ok=True)
            print(f"Created (or exists) repo: {args.repo_id}")
        except Exception as e:
            print(f"Warning: create_repo failed: {e}")

    if not os.path.isdir(args.source_dir):
        print(f"ERROR: source directory '{args.source_dir}' not found")
        sys.exit(1)

    files = list(find_files(args.source_dir, exts=args.ext))
    if not files:
        print(f"No files found in {args.source_dir} matching extensions {args.ext}")
        sys.exit(0)

    print(f"Found {len(files)} files to upload. Starting uploads...")

    for fp in files:
        rel = os.path.relpath(fp, args.source_dir)
        if args.path_prefix:
            path_in_repo = os.path.join(args.path_prefix, rel).replace("\\", "/")
        else:
            path_in_repo = rel.replace("\\", "/")

        print(f"Uploading {fp} -> {args.repo_id}/{path_in_repo}")
        try:
            api.upload_file(
                path_or_fileobj=fp,
                path_in_repo=path_in_repo,
                repo_id=args.repo_id,
                token=token,
            )
        except Exception as e:
            print(f"Failed to upload {fp}: {e}")

    print("Upload complete.")


if __name__ == "__main__":
    main()
