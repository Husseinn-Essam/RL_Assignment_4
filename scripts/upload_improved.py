import argparse
import os
import sys
import yaml
from huggingface_hub import HfApi
from dotenv import load_dotenv

# Extensions to automatically upload (Code + Models + Configs)
ALLOWED_EXTENSIONS = {".py", ".pt", ".pth", ".json", ".yaml", ".yml", ".mp4"}


def find_files(source_dir):
    """Recursively find all relevant files in the source directory."""
    files_to_upload = []
    for root, _, filenames in os.walk(source_dir):
        for fn in filenames:
            ext = os.path.splitext(fn)[1].lower()
            if ext in ALLOWED_EXTENSIONS:
                full_path = os.path.join(root, fn)
                files_to_upload.append(full_path)
    return files_to_upload


def create_readme_content(repo_id, env_name, algorithm, mean_reward, hyperparams):
    """
    Generates a README (model card) including the YAML metadata block required for the HF leaderboard.
    """
    metadata = {
        "tags": [
            "deep-reinforcement-learning",
            "reinforcement-learning",
            "deep-rl-course",
            env_name,
            "custom-implementation",
        ],
        "library_name": "pytorch",
        "env_id": env_name,
        "model-index": [
            {
                "name": repo_id,
                "results": [
                    {
                        "task": {"type": "reinforcement-learning", "name": "Reinforcement Learning"},
                        "dataset": {"name": env_name, "type": env_name},
                        "metrics": [{"type": "mean_reward", "value": float(mean_reward), "name": "Mean Reward"}],
                    }
                ],
            }
        ],
    }

    # Convert metadata to YAML string (leaderboard expects the YAML header)
    yaml_block = yaml.dump(metadata, default_flow_style=False, sort_keys=False)

    readme = f"""---
{yaml_block}---

# {algorithm} Agent for {env_name}

This is a custom implementation of **{algorithm}** trained on the **{env_name}** environment.

## Performance
- **Mean Reward**: {mean_reward}

## Usage

To load this agent, ensure the repository contains the source files referenced below (included in this upload):

```python
import torch
import gymnasium as gym
from gym_environment import create_environment, CarRacingActionWrapper, CarRacingPreprocessor
from models import TD3Agent

# 1. Setup Environment
env = create_environment('{env_name}', render_mode='human')

# 2. Initialize Agent
# (adjust state/action sizes to your implementation)
# agent = TD3Agent(state_size, action_size, device='cpu')

# 3. Load Model
# checkpoint = torch.load('model.pt', map_location='cpu')
# agent.actor.load_state_dict(checkpoint['actor_state_dict'])

# 4. Run
# obs, _ = env.reset()
# done = False
# while not done:
#     action = agent.select_action(obs, epsilon=0)
#     obs, reward, terminated, truncated, _ = env.step(action)
#     done = terminated or truncated
# env.close()
```

## Hyperparameters

```yaml
{yaml.safe_dump(hyperparams, sort_keys=False)}
```

"""

    return readme


def main():
    parser = argparse.ArgumentParser(description="Smart Uploader for RL Leaderboard")
    parser.add_argument("--repo-id", required=True, help="Hugging Face repo ID (e.g., username/td3-carracing)")
    parser.add_argument("--model-file", required=True, help="Path to your best .pt model file")
    parser.add_argument("--mean-reward", type=float, required=True, help="The mean reward achieved (e.g., 722.20)")
    parser.add_argument("--env-name", default="CarRacing-v3", help="Environment ID")
    parser.add_argument("--algo", default="TD3", help="Algorithm Name")
    parser.add_argument("--token", help="HF Write Token (optional if logged in via CLI)")
    parser.add_argument("--video", help="Path to a demo video (mp4) to upload")
    parser.add_argument("--source-dir", default=".", help="Directory to search for source files (default: current dir)")

    args = parser.parse_args()

    # 1. Authentication
    load_dotenv()
    token = args.token or os.environ.get("HF_TOKEN") or os.environ.get("HF_TOKEN_WRITE") or os.environ.get("HF_WRITE_TOKEN")
    api = HfApi(token=token) if token else HfApi()

    print(f"🚀 Preparing to upload to: {args.repo_id}")

    # 2. Create Repository (if it doesn't exist)
    try:
        repo_url = api.create_repo(repo_id=args.repo_id, exist_ok=True, private=False, token=token)
        print(f"✅ Repo ready: {repo_url}")
    except Exception as e:
        print(f"⚠️ Repo creation/check failed: {e}")

    # 3. Gather & Upload Files
    print("📦 Gathering files...")

    # A. Upload Model Weights (rename to model.pt in the repo)
    if not os.path.exists(args.model_file):
        print(f"❌ Model file not found: {args.model_file}")
        sys.exit(1)

    try:
        api.upload_file(
            path_or_fileobj=args.model_file,
            path_in_repo="model.pt",
            repo_id=args.repo_id,
            commit_message=f"Upload best model (Reward: {args.mean_reward})",
            token=token,
        )
        print(f"   - Uploaded model weights: {args.model_file}")
    except Exception as e:
        print(f"❌ Failed to upload model: {e}")

    # B. Upload relevant source files (search source-dir)
    source_files = find_files(args.source_dir)
    # Prefer a handful of critical files if present
    priority_files = ["gym_environment.py", "models.py", "td3_agent.py", "buffers.py", "networks.py"]

    uploaded = set()
    for pf in priority_files:
        if os.path.exists(pf):
            try:
                api.upload_file(path_or_fileobj=pf, path_in_repo=pf, repo_id=args.repo_id, token=token)
                print(f"   - Uploaded source: {pf}")
                uploaded.add(os.path.abspath(pf))
            except Exception as e:
                print(f"   ⚠️ Failed to upload {pf}: {e}")

    # Upload other discovered files (avoid duplicates)
    for path in source_files:
        abspath = os.path.abspath(path)
        if abspath in uploaded:
            continue
        repo_path = os.path.relpath(path, start=args.source_dir).replace('\\', '/')
        try:
            api.upload_file(path_or_fileobj=path, path_in_repo=repo_path, repo_id=args.repo_id, token=token)
            print(f"   - Uploaded: {repo_path}")
        except Exception as e:
            print(f"   ⚠️ Failed to upload {path}: {e}")

    # C. Demo Video (Optional)
    if args.video:
        if os.path.exists(args.video):
            try:
                api.upload_file(path_or_fileobj=args.video, path_in_repo="replay.mp4", repo_id=args.repo_id, token=token)
                print(f"   - Uploaded video replay: {args.video}")
            except Exception as e:
                print(f"   ⚠️ Failed to upload video: {e}")
        else:
            print(f"   ⚠️ Video not found: {args.video}")

    # 4. Generate and Upload README (Model Card)
    hyperparams = {
        "env_name": args.env_name,
        "algorithm": args.algo,
        "gamma": 0.99,
        "tau": 0.005,
        "noise_type": "Ornstein-Uhlenbeck",
        "batch_size": 256,
        "buffer_size": 1000000,
    }

    readme_content = create_readme_content(args.repo_id, args.env_name, args.algo, args.mean_reward, hyperparams)

    try:
        api.upload_file(
            path_or_fileobj=readme_content.encode("utf-8"),
            path_in_repo="README.md",
            repo_id=args.repo_id,
            commit_message="Update Model Card with Leaderboard Metrics",
            token=token,
        )
        print("✅ Uploaded README.md with Leaderboard tags")
    except Exception as e:
        print(f"❌ Failed to upload README.md: {e}")

    print("\n🎉 Success! Your model is live and indexed (if uploads succeeded).")
    print(f"🔗 View here: https://huggingface.co/{args.repo_id}")


if __name__ == "__main__":
    main()
