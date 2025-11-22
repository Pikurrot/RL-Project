from __future__ import annotations
import gymnasium as gym
import ale_py
from pathlib import Path
from stable_baselines3.common.vec_env import VecFrameStack, VecVideoRecorder
from stable_baselines3.common.env_util import make_vec_env as sb3_make_vec_env

gym.register_envs(ale_py)


def make_env(
	config: dict,
) -> gym.Env:
	# TODO: Put wrappers here
	env = gym.make(config["env"]["env_id"], render_mode="rgb_array")
	return env

def make_vec_env(
	config: dict,
) -> gym.Env:
	# Create a real VecEnv with n_envs
	venv = sb3_make_vec_env(
		lambda: make_env(config),
		n_envs=config["env"]["n_envs"],
		monitor_dir=config["monitor_dir"],
	)
	# Frame-stack observations (default to 4 if not provided)
	n_stack = config["env"].get("frame_stack", 4)
	venv = VecFrameStack(venv, n_stack=n_stack)
	if config["video_dir"] is not None:
		save_name = config["wandb"]["run_name"]
		video_dir = Path(config["video_dir"]) / save_name
		i = 1
		while video_dir.exists():
			video_dir = video_dir.parent / f"{save_name}_{i}"
			i += 1
		video_dir.mkdir(parents=True, exist_ok=True)
		venv = VecVideoRecorder(
			venv,
			video_folder=video_dir,
			record_video_trigger=lambda x: x % config["env"]["video_every"] == 0,
			video_length=config["env"]["video_length"],
		)
	return venv
