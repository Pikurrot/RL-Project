from __future__ import annotations
import torch
import torch.nn as nn
import gymnasium as gym
from pathlib import Path
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


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


# Adapted from: https://stable-baselines3.readthedocs.io/en/master/guide/custom_policy.html
class CustomCNN(BaseFeaturesExtractor):
	def __init__(
		self,
		observation_space: gym.spaces.Box,
		features_dim: int = 128,
		checkpoint_path: str = None,
	):
		super().__init__(observation_space, features_dim)
		# We assume CxHxW images (channels first)
        # Re-ordering will be done by pre-preprocessing or wrapper
		n_input_channels = observation_space.shape[0]
		self.cnn = nn.Sequential(
			nn.Conv2d(n_input_channels, 64, kernel_size=8, stride=4, padding=0),
			nn.ReLU(),
			nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=0),
			nn.ReLU(),
			nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
			nn.ReLU(),
			nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
			nn.ReLU(),
			nn.Flatten(),
		)

		# Compute shape by doing one forward pass
		with torch.no_grad():
			n_flatten = self.cnn(
				torch.as_tensor(observation_space.sample()[None]).float()
			).shape[1]
		self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())

		try:
			if checkpoint_path is not None:
				print(f"Loading checkpoint from: {checkpoint_path}")
				device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
				ckpt = torch.load(checkpoint_path, map_location=device)
				state = ckpt.get("extractor_state_dict", ckpt)
				missing, unexpected = self.load_state_dict(state, strict=False)
				if missing:
					print(f"Missing keys: {missing}")
				if unexpected:
					print(f"Unexpected keys: {unexpected}")
		except FileNotFoundError:
			print(f"Checkpoint not found: {checkpoint_path}. Loading from './'")
			checkpoint_path = "./pretrained_cnn.pt"
			ckpt = torch.load(checkpoint_path, map_location=device)
			state = ckpt.get("extractor_state_dict", ckpt)
			missing, unexpected = self.load_state_dict(state, strict=False)
			if missing:
				print(f"Missing keys: {missing}")
			if unexpected:
				print(f"Unexpected keys: {unexpected}")

	def forward(self, observations: torch.Tensor) -> torch.Tensor:
		return self.linear(self.cnn(observations))
