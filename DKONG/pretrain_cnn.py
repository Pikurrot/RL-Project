from __future__ import annotations
import os
import random
import socket
HOSTNAME = socket.gethostname()

if HOSTNAME == "cudahpc16":
	# idk who set up this cluster but without this the gpu is not detected
	os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
	os.environ["CUDA_VISIBLE_DEVICES"] = "6"
	os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import random
from collections import deque
from pathlib import Path
from typing import Callable, Deque, Dict, List

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import wandb

from environment.env import make_env
from models.base import CustomCNN
from _logging import load_config, load_pretrain_config

def prepare_run_dir(base_dir: Path, run_name: str) -> Path:
	target_dir = Path(base_dir) / run_name
	i = 1
	while target_dir.exists():
		target_dir = Path(base_dir) / f"{run_name}_{i}"
		i += 1
	target_dir.mkdir(parents=True, exist_ok=True)
	return target_dir


def create_stacked_space(
	observation_space: gym.spaces.Box,
	frame_stack: int,
) -> gym.spaces.Box:
	# Create a space like (frame_stack * C, H, W)
	c, h, w = observation_space.shape
	stacked_shape = (frame_stack * c, h, w)
	low = np.repeat(observation_space.low, frame_stack, axis=0)
	high = np.repeat(observation_space.high, frame_stack, axis=0)
	return gym.spaces.Box(low=low, high=high, dtype=observation_space.dtype, shape=stacked_shape)


def stack_frames(frames: Deque[np.ndarray]) -> np.ndarray:
	return np.concatenate(list(frames), axis=0)


def bootstrap_stack(
	env: gym.Env,
	stack: Deque[np.ndarray],
	frame_stack: int,
	reset_kwargs: dict,
	step_action: Callable[[], int],
	reset_action_policy: Callable[[], None],
) -> None:
	# Clear the stack and reset the action policy
	stack.clear()
	reset_action_policy()
	obs, _ = env.reset(**reset_kwargs)
	stack.append(obs.copy())
	while len(stack) < frame_stack:
		action = step_action()
		next_obs, _, terminated, truncated, _ = env.step(action)
		if terminated or truncated:
			reset_action_policy()
			obs, _ = env.reset(**reset_kwargs)
			stack.clear()
			stack.append(obs.copy())
			continue
		stack.append(next_obs.copy())


class InverseDynamicsHead(nn.Module):
	def __init__(self, input_dim: int, hidden_dim: int, n_actions: int):
		super().__init__()
		self.net = nn.Sequential(
			nn.Linear(input_dim, hidden_dim),
			nn.ReLU(),
			nn.Linear(hidden_dim, n_actions),
		)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		return self.net(x)


def to_tensor(obs: np.ndarray, device: torch.device) -> torch.Tensor:
	return torch.from_numpy(obs).unsqueeze(0).to(device=device, dtype=torch.float32)


def save_transition_chunk(
	chunk: List[Dict[str, np.ndarray]],
	target_dir: Path,
	chunk_idx: int,
) -> None:
	if not chunk:
		return
	target_dir.mkdir(parents=True, exist_ok=True)
	data = {
		"s_t": np.stack([entry["s_t"] for entry in chunk]),
		"s_tp1": np.stack([entry["s_tp1"] for entry in chunk]),
		"action": np.array([entry["action"] for entry in chunk], dtype=np.int64),
	}
	path = target_dir / f"chunk_{chunk_idx:05d}.npz"
	np.savez_compressed(path, **data)


def save_checkpoint(
	step: int,
	extractor: CustomCNN,
	head: InverseDynamicsHead,
	optimizer: optim.Optimizer,
	target_dir: Path,
) -> None:
	target_dir.mkdir(parents=True, exist_ok=True)
	ckpt = {
		"step": step,
		"extractor_state_dict": extractor.state_dict(),
		"head_state_dict": head.state_dict(),
		"optimizer_state_dict": optimizer.state_dict(),
	}
	path = target_dir / f"step_{step:07d}.pt"
	torch.save(ckpt, path)


def init_wandb(config: dict) -> wandb.Run:
	wandb_cfg = config["wandb"].copy()
	entity = wandb_cfg.pop("entity")
	project = wandb_cfg.pop("project")
	run_name = wandb_cfg.pop("run_name")
	return wandb.init(
		entity=entity,
		project=project,
		name=run_name,
		config=config,
	)


def main() -> None:
	main_config = load_config()
	pretrain_config = load_pretrain_config()
	pre_cfg = pretrain_config["pretrain"]

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	print(f"Using device: {device}")

	random_seed = pre_cfg.get("seed")
	if random_seed is not None:
		random.seed(random_seed)
		np.random.seed(random_seed)
		torch.manual_seed(random_seed)

	env = make_env(main_config)
	frame_stack = int(main_config["env"]["frame_stack"])
	stacked_space = create_stacked_space(env.observation_space, frame_stack)

	features_kwargs = (
		main_config["model"]["PPO"]["policy_kwargs"].get("features_extractor_kwargs", {})
	)
	features_dim = int(features_kwargs.get("features_dim", 256))
	extractor = CustomCNN(stacked_space, features_dim=features_dim).to(device)

	head_hidden_dim = int(pre_cfg.get("head_hidden_dim", 512))
	n_actions = env.action_space.n
	head = InverseDynamicsHead(features_dim * 2, head_hidden_dim, n_actions).to(device)

	optimizer = optim.Adam(
		list(extractor.parameters()) + list(head.parameters()),
		lr=float(pre_cfg.get("learning_rate", 1e-4)),
	)
	criterion = nn.CrossEntropyLoss()

	run_dir = prepare_run_dir(Path(main_config["checkpoints_dir_pretrained"]), pretrain_config["wandb"]["run_name"])
	checkpoint_dir = run_dir / "checkpoints"
	checkpoint_dir.mkdir(parents=True, exist_ok=True)
	save_dataset = bool(pre_cfg.get("save_dataset", True))
	dataset_chunk_size = max(1, int(pre_cfg.get("dataset_chunk_size", 2048))) if save_dataset else 0
	dataset_dir = None
	if save_dataset:
		dataset_dir = run_dir / "dataset"
		dataset_dir.mkdir(parents=True, exist_ok=True)
	print(f"[Pretrain] Saving checkpoints to: {checkpoint_dir}")
	if dataset_dir is not None:
		print(f"[Pretrain] Saving dataset chunks to: {dataset_dir}")
	wandb_run = init_wandb(pretrain_config)

	total_steps = int(pre_cfg["total_steps"])
	save_every = int(pre_cfg["save_every_steps"])
	log_interval = int(pre_cfg["log_interval"])

	step = 0
	running_loss = 0.0
	running_updates = 0
	running_correct = 0
	running_samples = 0

	stack: Deque[np.ndarray] = deque(maxlen=frame_stack)
	dataset_chunk: List[Dict[str, np.ndarray]] = []
	chunk_idx = 0
	reset_kwargs = {}
	if random_seed is not None:
		reset_kwargs["seed"] = random_seed

	action_repeat_steps = max(1, int(pre_cfg.get("action_repeat_steps", frame_stack)))
	minimal_actions = main_config["env"]["minimal_actions"]
	action_lookup = {value: idx for idx, value in enumerate(minimal_actions)}
	current_action_idx = 0
	action_repeat_remaining = 0

	def reset_action_policy() -> None:
		# Reset the action repeat counter
		nonlocal action_repeat_remaining
		action_repeat_remaining = 0

	def ensure_action_sampled() -> None:
		# Sample a new action if the action repeat counter is 0
		nonlocal current_action_idx, action_repeat_remaining
		if action_repeat_remaining <= 0:
			chosen_action = random.choice(minimal_actions)
			current_action_idx = action_lookup[chosen_action]
			action_repeat_remaining = action_repeat_steps

	def step_action() -> int:
		# Step the action repeat counter and return the current action index
		nonlocal action_repeat_remaining
		ensure_action_sampled()
		action_repeat_remaining -= 1
		return current_action_idx

	bootstrap_stack(env, stack, frame_stack, reset_kwargs, step_action, reset_action_policy)

	try:
		while step < total_steps:
			current_stack = stack_frames(stack).astype(np.float32)
			ensure_action_sampled()
			action_label = current_action_idx

			next_frames: List[np.ndarray] = []
			valid_transition = True
			for _ in range(frame_stack):
				action = step_action()
				next_obs, _, terminated, truncated, _ = env.step(action)
				next_frames.append(next_obs.copy())
				if terminated or truncated:
					valid_transition = False
					bootstrap_stack(env, stack, frame_stack, reset_kwargs, step_action, reset_action_policy)
					break
				stack.append(next_obs.copy())

			if not valid_transition or len(next_frames) < frame_stack:
				continue

			next_stack = np.concatenate(next_frames, axis=0).astype(np.float32)

			if dataset_dir is not None:
				dataset_chunk.append(
					{
						"s_t": current_stack.copy(),
						"s_tp1": next_stack.copy(),
						"action": action_label,
					}
				)
				if len(dataset_chunk) >= dataset_chunk_size:
					save_transition_chunk(dataset_chunk, dataset_dir, chunk_idx)
					chunk_idx += 1
					dataset_chunk.clear()

			# Training step
			z_t = extractor(to_tensor(current_stack, device))
			z_tp1 = extractor(to_tensor(next_stack, device))
			features = torch.cat([z_t, z_tp1], dim=1)
			logits = head(features)
			target_action = torch.tensor([action_label], dtype=torch.long, device=device)
			loss = criterion(logits, target_action)

			optimizer.zero_grad()
			loss.backward()
			optimizer.step()

			with torch.no_grad():
				pred_action = logits.argmax(dim=1)
				is_correct = (pred_action == target_action).sum().item()

			running_loss += loss.item()
			running_updates += 1
			running_correct += is_correct
			running_samples += target_action.size(0)

			step += 1

			if step % log_interval == 0 or step == 1:
				avg_loss = running_loss / max(running_updates, 1)
				avg_acc = running_correct / max(running_samples, 1)
				print(f"[Pretrain] Step {step}/{total_steps} | Loss: {avg_loss:.6f} | Acc: {avg_acc:.3f}")
				if wandb_run is not None:
					wandb.log(
						{
							"loss": avg_loss,
							"accuracy": avg_acc,
							"step": step,
						}
					)
				running_loss = 0.0
				running_updates = 0
				running_correct = 0
				running_samples = 0

			if step % save_every == 0 or step == total_steps:
				save_checkpoint(step, extractor, head, optimizer, checkpoint_dir)

	except KeyboardInterrupt:
		print("[Pretrain] Interrupted by user. Saving progress...")
	finally:
		# Flush dataset and final checkpoint
		if dataset_dir is not None and dataset_chunk:
			save_transition_chunk(dataset_chunk, dataset_dir, chunk_idx)
		save_checkpoint(step, extractor, head, optimizer, checkpoint_dir)
		env.close()
		if wandb_run is not None:
			wandb_run.finish()
		print(f"[Pretrain] Finished at step {step}.")


if __name__ == "__main__":
	main()

