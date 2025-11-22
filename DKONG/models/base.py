from __future__ import annotations
from pathlib import Path
from stable_baselines3.common.base_class import BaseAlgorithm

class BaseModel:
	def __init__(self, config: dict):
		self.config = config
		# check if path exists, and if so, change to path_i
		save_name = self.config["wandb"]["run_name"]
		self.checkpoints_dir = Path(config["checkpoints_dir"]) / save_name
		i = 1
		while self.checkpoints_dir.exists():
			self.checkpoints_dir = self.checkpoints_dir.parent / f"{save_name}_{i}"
			i += 1
		self.checkpoints_dir.mkdir(parents=True, exist_ok=True)
