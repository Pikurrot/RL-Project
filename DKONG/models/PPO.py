from __future__ import annotations
import torch
import gymnasium as gym
from stable_baselines3 import PPO as PPOsb3
from pathlib import Path

from _logging import build_callbacks
from models.base import BaseModel


class PPO(BaseModel, PPOsb3):
	def __init__(self, config: dict, vec_env: gym.Env, device: torch.device = None):
		BaseModel.__init__(self, config)
		self.ppo_config = config["model"]["PPO"]
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
			ent_coef=self.ppo_config["ent_coef"],
			vf_coef=self.ppo_config["vf_coef"],
			tensorboard_log=str(Path(self.config["monitor_dir"]) / "tb"),
			verbose=1,
			device=device
		)
		self.total_timesteps = self.ppo_config["total_timesteps"]

	def learn(
		self,
		progress_bar: bool = True
	):
		callbacks = build_callbacks(self.config, self.checkpoints_dir)
		super().learn(
			progress_bar=progress_bar,
			total_timesteps=self.total_timesteps,
			callback=callbacks,
		)
