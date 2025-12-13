from __future__ import annotations

import sys
from pathlib import Path

import torch
from stable_baselines3 import DQN as SB3DQN
from stable_baselines3 import PPO as SB3PPO

from _logging import load_config
from environment.env import make_eval_vec_env
# Importing CustomCNN registers the class so SB3 can unpickle models
from models.base import CustomCNN  # noqa: F401

# --- User options ---------------------------------------------------------
# Path to the checkpoint to load. If relative, it is resolved under
# config["checkpoints_dir"].
CHECKPOINT_OVERRIDE = "ppo_exp_all_rewards_cancelled/best/best_model.zip"

# Number of environment steps (frames after frame stacking) to record.
VIDEO_STEPS = 2_000

# Override where videos are written. If None, uses config["video_dir"].
# If relative, it is resolved relative to the project root.
VIDEO_DIR_OVERRIDE: str | None = "/home/elopez/RL-Project/DKONG/videos/custom"

# Override the run_name used for the video directory. This also controls the
# subfolder created under video_dir. If None, uses the config run_name with
# RUN_NAME_SUFFIX appended.
VIDEO_RUN_NAME_OVERRIDE: str | None = CHECKPOINT_OVERRIDE.split("/")[0] + "_video"

# Override the video file prefix (default: "eval-video"). Useful to control
# the final filename within the video directory.
VIDEO_NAME_PREFIX: str | None = None  # e.g., "donkeykong-play"

# Suffix added to the config run_name so the video is saved in a fresh folder
# when VIDEO_RUN_NAME_OVERRIDE is not set.
RUN_NAME_SUFFIX = "video"
# --------------------------------------------------------------------------


def _resolve_checkpoint(config: dict) -> Path:
    override = Path(CHECKPOINT_OVERRIDE)
    if override.is_absolute():
        return override
    return Path(config["checkpoints_dir"]) / override


def _print_video_destination(env) -> None:
    # VecVideoRecorder exposes video_path after recording starts; otherwise
    # fall back to the folder so the user knows where to look.
    video_path = getattr(env, "video_path", None)
    video_folder = getattr(env, "video_folder", None)
    if video_path:
        print(f"Video written to: {video_path}")
    elif video_folder:
        print(f"Video folder: {video_folder}")
    else:
        print("Video saved (path unavailable on wrapper).")


def main() -> None:
    config = load_config()

    # Override config with script-specific choices
    config["model"]["PPO"]["checkpoint_path"] = CHECKPOINT_OVERRIDE
    config["eval"]["video_length"] = int(VIDEO_STEPS)
    if VIDEO_RUN_NAME_OVERRIDE:
        config["wandb"]["run_name"] = VIDEO_RUN_NAME_OVERRIDE
    else:
        config["wandb"]["run_name"] = f"{config['wandb']['run_name']}_{RUN_NAME_SUFFIX}"

    if VIDEO_DIR_OVERRIDE:
        video_dir = Path(VIDEO_DIR_OVERRIDE)
        if not video_dir.is_absolute():
            video_dir = Path(__file__).resolve().parent / video_dir
        video_dir.mkdir(parents=True, exist_ok=True)
        config["video_dir"] = str(video_dir)

    checkpoint_path = _resolve_checkpoint(config)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    # Build evaluation env with video recording enabled
    eval_env = make_eval_vec_env(config)
    if VIDEO_NAME_PREFIX and hasattr(eval_env, "name_prefix"):
        eval_env.name_prefix = VIDEO_NAME_PREFIX
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Loaded checkpoint: {checkpoint_path}")

    # Load the model
    model_name = config["model"]["select_model"].upper()
    if model_name == "PPO":
        model = SB3PPO.load(checkpoint_path, env=eval_env, device=device)
    elif model_name == "DQN":
        model = SB3DQN.load(checkpoint_path, env=eval_env, device=device)
    else:
        raise ValueError(f"Unsupported model type: {model_name}")

    # Rollout and record
    obs = eval_env.reset()
    steps = 0
    while steps < VIDEO_STEPS:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, dones, _ = eval_env.step(action)
        steps += 1
        if dones.any():
            obs = eval_env.reset()

    _print_video_destination(eval_env)
    eval_env.close()
    print(f"Completed {steps} steps.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit("Interrupted.")

