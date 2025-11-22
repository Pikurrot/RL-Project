from __future__ import annotations
import torch
import gymnasium as gym
from stable_baselines3 import DQN as DQNsb3
from pathlib import Path

from _logging import build_callbacks
from models.base import BaseModel


class DQN(BaseModel, DQNsb3):
	def __init__(self, config: dict, vec_env: gym.Env, device: torch.device = None):
		BaseModel.__init__(self, config)
		self.dqn_config = config["model"]["DQN"]
		DQNsb3.__init__(
			self,
			policy=self.dqn_config["policy"],
			env=vec_env,
			learning_rate=float(self.dqn_config["learning_rate"]),
			buffer_size=self.dqn_config["buffer_size"],
			learning_starts=self.dqn_config["learning_starts"],
			batch_size=self.dqn_config["batch_size"],
			tau=self.dqn_config["tau"],
			gamma=self.dqn_config["gamma"],
			train_freq=self.dqn_config["train_freq"],
			target_update_interval=self.dqn_config["target_update_interval"],
			exploration_fraction=self.dqn_config["exploration_fraction"],
			exploration_initial_eps=self.dqn_config["exploration_initial_eps"],
			exploration_final_eps=self.dqn_config["exploration_final_eps"],
			tensorboard_log=str(Path(self.config["monitor_dir"]) / "tb"),
			verbose=1,
			device=device
		)
		self.total_timesteps = self.dqn_config["total_timesteps"]

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
