from __future__ import annotations
import torch
import gymnasium as gym
from pathlib import Path
from stable_baselines3 import PPO as PPOsb3
from stable_baselines3.common.callbacks import BaseCallback, CallbackList

from _logging import build_callbacks
from .base import BaseModel, CustomCNN
from schedulers import build_sb3_schedule


class EntropyScheduleCallback(BaseCallback):
	def __init__(self, schedule_fn):
		super().__init__()
		self.schedule_fn = schedule_fn

	def _on_training_start(self) -> None:
		# set initial ent_coef based on start-of-training progress (1.0)
		current_ent = float(self.schedule_fn(1.0))
		self.model.ent_coef = current_ent
		self.logger.record("train/ent_coef", current_ent)

	def _on_rollout_start(self) -> bool:
		# update entropy coef based on current progress remaining
		current_ent = float(self.schedule_fn(self.model._current_progress_remaining))
		self.model.ent_coef = current_ent
		self.logger.record("train/ent_coef", current_ent)
		return True

	def _on_step(self) -> bool:
		return True


class PPO(BaseModel, PPOsb3):
	def __init__(self, config: dict, vec_env: gym.Env, device: torch.device = None):
		BaseModel.__init__(self, config)
		self.ppo_config = config["model"]["PPO"]
		self.policy_kwargs = self.ppo_config["policy_kwargs"]
		class_name = self.policy_kwargs["features_extractor_class"]
		if class_name == "CustomCNN":
			self.policy_kwargs["features_extractor_class"] = CustomCNN
			checkpoint_path = self.policy_kwargs["features_extractor_kwargs"]["checkpoint_path"]
			if checkpoint_path is not None:
				checkpoints_dir_pretrained = Path(config["checkpoints_dir_pretrained"])
				checkpoint_path = checkpoints_dir_pretrained / checkpoint_path
				self.policy_kwargs["features_extractor_kwargs"]["checkpoint_path"] = str(checkpoint_path)

		ent_spec = self.ppo_config.get("ent_coef_schedule")
		self.ent_coef_schedule_fn = None
		if ent_spec is not None:
			self.ent_coef_schedule_fn = build_sb3_schedule(ent_spec)
			ent_coef_init = float(self.ent_coef_schedule_fn(1.0))  # start of training
		else:
			ent_coef_init = float(self.ppo_config["ent_coef"])

		PPOsb3.__init__(
			self,
			policy=self.ppo_config["policy"],
			env=vec_env,
			learning_rate=float(self.ppo_config["learning_rate"]),
			n_steps=self.ppo_config["n_steps"],
			batch_size=self.ppo_config["batch_size"],
			n_epochs=self.ppo_config["n_epochs"],
			gamma=self.ppo_config["gamma"],
			clip_range=self.ppo_config["clip_range"],
			ent_coef=ent_coef_init,
			vf_coef=self.ppo_config["vf_coef"],
			tensorboard_log=str(Path(self.config["monitor_dir"]) / "tb"),
			verbose=1,
			device=device,
			policy_kwargs=self.policy_kwargs
		)
		self.total_timesteps = self.ppo_config["total_timesteps"]

	def learn(
		self,
		progress_bar: bool = True
	):
		callbacks = build_callbacks(self.config, self.checkpoints_dir)
		if self.ent_coef_schedule_fn is not None:
			entropy_cb = EntropyScheduleCallback(self.ent_coef_schedule_fn)
			if callbacks is None:
				callbacks = entropy_cb
			elif isinstance(callbacks, CallbackList):
				callbacks.callbacks.insert(0, entropy_cb)
			else:
				callbacks = CallbackList([entropy_cb, callbacks])

		super().learn(
			progress_bar=progress_bar,
			total_timesteps=self.total_timesteps,
			callback=callbacks,
		)
