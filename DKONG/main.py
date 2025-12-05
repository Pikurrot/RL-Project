from __future__ import annotations
import os
import socket
HOSTNAME = socket.gethostname()

if HOSTNAME == "cudahpc16":
	# idk who set up this cluster but without this the gpu is not detected
	os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
	os.environ["CUDA_VISIBLE_DEVICES"] = "3"
	os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch
import gymnasium as gym
import ale_py

from environment.env import make_vec_env
from models.DQN import DQN
from models.PPO import PPO
from _logging import init_wandb, load_config

gym.register_envs(ale_py)


if __name__ == "__main__":
	config = load_config()
	run = init_wandb(config)

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	print(f"Using device: {device}")
	
	vec_env = make_vec_env(config)
	if config["model"]["select_model"] == "DQN":
		model = DQN(config, vec_env, device=device)
	elif config["model"]["select_model"] == "PPO":
		model = PPO(config, vec_env, device=device)
	else:
		raise ValueError(f"Invalid model: {config['model']['select_model']}")

	model.learn(progress_bar=True)
	run.finish()
