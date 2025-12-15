import gymnasium as gym
import ale_py

# version
print("Using Gymnasium version {}".format(gym.__version__))
gym.register_envs(ale_py)

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import os
from collections import deque
import matplotlib.pyplot as plt
from gymnasium.wrappers import FrameStackObservation, AtariPreprocessing

print("PyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("CUDA version:", torch.version.cuda)
print("Number of GPUs:", torch.cuda.device_count())

SAVE_PATH = "checkpoints3"
os.makedirs(SAVE_PATH, exist_ok=True)

# ---------------------------
# Device
# ---------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ---------------------------
# Environment (Pong with reduced actions)
# ---------------------------
class PongReducedActions(gym.ActionWrapper):
    """
    Reduce Pong action space:
    0 = NOOP, 1 = UP (RIGHT), 2 = DOWN (LEFT)
    Automatically fires when needed.
    """
    def __init__(self, env):
        super().__init__(env)
        self.action_map = [0, 2, 3]   # mapping to original actions

    def action(self, act):
        return self.action_map[act]

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        while info.get("needs_reset", False):
            obs, _, _, _, info = self.env.step(1)
        return obs, info


ENV_NAME = "PongNoFrameskip-v4"
env = gym.make(ENV_NAME)
env = AtariPreprocessing(env, grayscale_obs=True, scale_obs=True, frame_skip=4)
env = PongReducedActions(env)
env = FrameStackObservation(env, 4)

n_actions = 3
print("Action space:", n_actions)
obs_shape = env.observation_space.shape
print("Observation shape:", obs_shape)

# ---------------------------
# Policy Network
# ---------------------------
class PolicyCNN(nn.Module):
    def __init__(self, n_actions):
        super().__init__()
        self.conv1 = nn.Conv2d(4, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)

        self.fc1 = nn.Linear(64 * 7 * 7, 512)
        self.fc2 = nn.Linear(512, n_actions)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.flatten(start_dim=1)
        x = F.relu(self.fc1(x))
        logits = self.fc2(x)
        return F.softmax(logits, dim=1)


model = PolicyCNN(n_actions).to(device)
log_file = open("training_pong_reduced_actions3.txt", "w")

# ---------------------------
# Optimizer
# ---------------------------
LEARNING_RATE = 2.5e-4
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# ---------------------------
# REINFORCE Parameters
# ---------------------------
GAMMA = 0.99
MAX_EPISODES = 1000
score_history = []

# ---------------------------
# Training Loop
# ---------------------------
for episode in range(MAX_EPISODES):
    state, _ = env.reset()
    done = False

    rewards = []
    log_probs = []
    entropies = []

    while not done:
        state_tensor = torch.from_numpy(np.array(state)).float().unsqueeze(0).to(device)

        probs = model(state_tensor)
        m = torch.distributions.Categorical(probs)
        action = m.sample()

        next_state, reward, terminated, truncated, _ = env.step(action.item())
        done = terminated or truncated

        # Save log prob + entropy
        log_probs.append(m.log_prob(action))
        entropies.append(m.entropy())

        # Reward clipping for stability
        rewards.append(np.sign(reward))

        state = next_state

    # Compute discounted returns
    G = 0
    returns = []
    for r in reversed(rewards):
        G = r + GAMMA * G
        returns.insert(0, G)
    returns = torch.tensor(returns, dtype=torch.float32).to(device)
    returns = (returns - returns.mean()) / (returns.std() + 1e-8)

    log_probs = torch.stack(log_probs)
    entropies = torch.stack(entropies)

    # Final loss (REINFORCE with entropy bonus)
    loss = -(log_probs * returns).sum() - 0.01 * entropies.sum()

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Logging
    episode_reward = sum(rewards)
    score_history.append(episode_reward)

    if (episode + 1) % 10 == 0:
        avg_reward = np.mean(score_history[-10:])
        print(f"Episode {episode+1}, Reward: {episode_reward}, Avg(10): {avg_reward:.2f}")
        log_file.write(f"Episode {episode+1}, Reward: {episode_reward}, Avg(10): {avg_reward:.2f}\n")
        log_file.flush()

    if (episode + 1) % 1000 == 0:
        ckpt = {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "episode": episode + 1,
            "avg_reward": avg_reward,
            "last_reward": episode_reward
        }
        checkpoint_path = os.path.join(SAVE_PATH, f"checkpoint_ep{episode+1}.pth")
        torch.save(ckpt, checkpoint_path)
        print(f"Checkpoint saved: {checkpoint_path}")

# Save final model
final_path = os.path.join(SAVE_PATH, "policy_final.pth")
torch.save(model.state_dict(), final_path)
print("\nTraining finished. Final model saved at:", final_path)
log_file.close()
