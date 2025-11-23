from __future__ import annotations
import logging
import threading
import ale_py
import gymnasium as gym
from pathlib import Path
from stable_baselines3.common.env_util import make_vec_env as sb3_make_vec_env
from stable_baselines3.common.vec_env import VecFrameStack, VecVideoRecorder

from .wrappers import (
	MinimalActionSpace,
	FireAtStart,
	AddChannelDim,
	ScaleObservation,
)

logger = logging.getLogger(__name__)
gym.register_envs(ale_py)


# Send video encoding to a background thread. It was taking too long and making training slow.
class AsyncVecVideoRecorder(VecVideoRecorder):
	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)
		self._encode_threads: list[threading.Thread] = []
		self._video_counter = 0

	def _stop_recording(self) -> None:
		# copy-paste from super source code, but using threading
		assert self.recording, "_stop_recording was called, but no recording was started"
		if len(self.recorded_frames) == 0:
			logger.warning("Ignored saving a video as there were zero frames to save.")
		else:
			frames = self.recorded_frames.copy()
			video_path = self.video_path
			fps = self.frames_per_sec

			def _encode() -> None:
				from moviepy.video.io.ImageSequenceClip import ImageSequenceClip

				clip = ImageSequenceClip(frames, fps=fps)
				clip.write_videofile(video_path, logger=None, audio=False)

			thread = threading.Thread(target=_encode, daemon=True)
			thread.start()
			self._encode_threads.append(thread)

		self.recorded_frames = []
		self.recording = False

	def _start_video_recorder(self) -> None:
		# give each video a unique prefix so repeated evaluations don't clobber files
		original_prefix = self.name_prefix
		self.name_prefix = f"{original_prefix}-{self._video_counter:04d}"
		self._video_counter += 1
		try:
			super()._start_video_recorder()
		finally:
			self.name_prefix = original_prefix

	def close(self) -> None:
		# join all threads before closing
		super().close()
		for thread in self._encode_threads:
			thread.join()
		self._encode_threads.clear()


def make_env(config: dict) -> gym.Env:
	# Create a single env. This is shared by training and evaluation.
	# TODO: Put wrappers here
	env = gym.make(config["env"]["env_id"], render_mode="rgb_array", obs_type="grayscale")
	env = FireAtStart(env)
	env = MinimalActionSpace(env, config)
	env = AddChannelDim(env)
	env = ScaleObservation(env)
	return env


def make_vec_env(
	config: dict,
) -> gym.Env:
	# Create a multi-env VecEnv for training
	venv = sb3_make_vec_env(
		lambda: make_env(config),
		n_envs=config["env"]["n_envs"],
		monitor_dir=config["monitor_dir"],
	)
	n_stack = config["env"]["frame_stack"]
	venv = VecFrameStack(venv, n_stack=n_stack)
	return venv


def make_eval_vec_env(
	config: dict,
) -> gym.Env:
	# Create a single env VecEnv for evaluation
	eval_env = sb3_make_vec_env(
		lambda: make_env(config),
		n_envs=1,
		monitor_dir=config["monitor_dir"],
	)
	eval_env = VecFrameStack(eval_env, n_stack=config["env"]["frame_stack"])

	if config["video_dir"] is not None:
		video_dir = _prepare_video_dir(config, suffix="eval")
		record_flag = {"record": True}

		def _record_video_trigger(step: int, flag=record_flag) -> bool:
			return flag["record"] and step == 0

		video_length = config["eval"]["video_length"]
		eval_env = AsyncVecVideoRecorder(
			eval_env,
			video_folder=str(video_dir),
			record_video_trigger=_record_video_trigger,
			video_length=video_length,
			name_prefix="eval-video",
		)
		setattr(eval_env, "_record_flag", record_flag)

	return eval_env


def _prepare_video_dir(config: dict, suffix: str) -> Path:
	# create video dir if it doesn't exist, and add _i to avoid overwriting
	save_name = config["wandb"]["run_name"]
	base_dir = Path(config["video_dir"])
	target_dir = base_dir / f"{save_name}_{suffix}"
	i = 1
	while target_dir.exists():
		target_dir = base_dir / f"{save_name}_{suffix}_{i}"
		i += 1
	target_dir.mkdir(parents=True, exist_ok=True)
	return target_dir
