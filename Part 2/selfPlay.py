import os
import numpy as np
import imageio
from typing import Dict
import random

import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
import torch
import torch.nn as nn
import torch.nn.functional as F

import supersuit as ss
from pettingzoo.atari import pong_v3

# ---------------------------
# ENVIRONMENT FACTORY
# ---------------------------
def make_base_env(render_mode=None):
    env = pong_v3.env(num_players=2, render_mode=render_mode)
    env = ss.sticky_actions_v0(env, repeat_action_probability=0.25)
    env = ss.color_reduction_v0(env, mode="B")
    env = ss.resize_v1(env, 84, 84)
    env = ss.frame_stack_v1(env, 4)
    env = ss.dtype_v0(env, dtype=np.uint8)
    env = ss.reshape_v0(env, (4, 84, 84))
    return env

# ---------------------------
# SINGLE-AGENT WRAPPER
# ---------------------------
class SingleAgentWrapper(gym.Env):
    """Wrap PettingZoo Pong so SB3 controls one agent; opponent is fixed."""
    def __init__(self, player_id, opponent_policy, render_mode=None):
        super().__init__()
        self.player_id = player_id
        self.opponent_id = "second_0" if player_id == "first_0" else "first_0"
        self.env = make_base_env(render_mode=render_mode)
        obs = self.env.observe(player_id)
        self.observation_space = spaces.Box(low=0, high=255, shape=obs.shape, dtype=np.uint8)
        self.action_space = self.env.action_space(player_id)
        self.opponent_policy = opponent_policy
        self.agent_iter = None

    def reset(self):
        obs_dict = self.env.reset()
        self.agent_iter = iter(self.env.agent_iter())
        obs_agent = None

        while True:
            agent = next(self.agent_iter)
            obs, reward, terminated, truncated, info = self.env.last()
            if agent == self.player_id:
                obs_agent = obs
                self.env.step(0)  # dummy step
                break
            else:
                act, _ = self.opponent_policy.predict(obs, deterministic=True)
                self.env.step(int(act[0]))
        return obs_agent, {}

    def step(self, action):
        obs_agent, reward_agent, done = None, 0, False
        while True:
            agent = next(self.agent_iter)
            obs, reward, terminated, truncated, info = self.env.last()
            if agent == self.player_id:
                obs_agent = obs
                reward_agent = reward
                done = terminated or truncated
                self.env.step(0)  # dummy step
                break
            else:
                act, _ = self.opponent_policy.predict(obs, deterministic=True)
                self.env.step(int(act[0]))
        return obs_agent, reward_agent, done, False, {}

    def render(self):
        return self.env.render()

    def close(self):
        self.env.close()

# ---------------------------
# DQN OPPONENT (optional)
# ---------------------------
class DQN(nn.Module):
    def __init__(self, in_channels, n_actions):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.fc1 = nn.Linear(64 * 7 * 7, 512)
        self.out = nn.Linear(512, n_actions)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        return self.out(x)

# ---------------------------
# RANDOM OPPONENT
# ---------------------------
class RandomPolicy:
    def __init__(self, action_space):
        self.action_space = action_space
    def predict(self, obs, deterministic=True):
        return np.array([self.action_space.sample()]), None

# ---------------------------
# EVALUATION / RECORDING
# ---------------------------
def evaluate_policies(left_model, right_model, episodes=3):
    env = make_base_env(render_mode=None)
    total_left, total_right = 0, 0
    for _ in range(episodes):
        obs_dict = env.reset()
        done = {a: False for a in env.agents}
        truncated = {a: False for a in env.agents}
        while not any(done.values()) and not any(truncated.values()):
            act_left, _ = left_model.predict(obs_dict["second_0"], deterministic=True)
            act_right, _ = right_model.predict(obs_dict["first_0"], deterministic=True)
            actions = {"second_0": int(act_left[0]), "first_0": int(act_right[0])}
            obs_dict, rew, done, truncated, _ = env.step(actions)
            total_left += rew["second_0"]
            total_right += rew["first_0"]
    env.close()
    return {"left_avg_reward": total_left/episodes, "right_avg_reward": total_right/episodes}

def record_match_to_gif(left_model, right_model, filename="match.gif", max_steps=1000):
    env = make_base_env(render_mode="rgb_array")
    obs = env.reset()
    frames = []
    for _ in range(max_steps):
        frames.append(env.render())
        act_left, _ = left_model.predict(obs["second_0"], deterministic=True)
        act_right, _ = right_model.predict(obs["first_0"], deterministic=True)
        actions = {"second_0": int(act_left[0]), "first_0": int(act_right[0])}
        obs, rew, done, truncated, _ = env.step(actions)
        if any(done.values()) or any(truncated.values()):
            break
    env.close()
    imageio.mimsave(filename, frames, fps=30)
    print(f"Saved GIF: {filename}")

# ---------------------------
# SELF-PLAY TRAINING
# ---------------------------
def self_play_training(total_iterations=1, train_timesteps=50000, eval_episodes=3):
    os.makedirs("selfplay_models", exist_ok=True)

    # initial opponent: random
    base_env = make_base_env()
    opponent = RandomPolicy(base_env.action_space("first_0"))
    base_env.close()

    # Initialize left and right PPO policies
    left_model = PPO("CnnPolicy", SingleAgentWrapper("second_0", opponent), verbose=1)
    right_model = PPO("CnnPolicy", SingleAgentWrapper("first_0", opponent), verbose=1)

    for it in range(total_iterations):
        print(f"\n=== ITERATION {it}: Train LEFT vs opponent ===")
        left_model.set_env(SingleAgentWrapper("second_0", opponent))
        left_model.learn(total_timesteps=train_timesteps)
        left_path = f"selfplay_models/left_iter_{it}.zip"
        left_model.save(left_path)
        opponent = PPO.load(left_path)  # freeze as opponent

        print(f"=== ITERATION {it}: Train RIGHT vs LEFT ===")
        right_model.set_env(SingleAgentWrapper("first_0", opponent))
        right_model.learn(total_timesteps=train_timesteps)
        right_path = f"selfplay_models/right_iter_{it}.zip"
        right_model.save(right_path)

        print(f"=== ITERATION {it}: Evaluation ===")
        eval_stats = evaluate_policies(PPO.load(left_path), PPO.load(right_path), episodes=eval_episodes)
        print("Eval:", eval_stats)

        gif_path = f"selfplay_models/match_iter_{it}.gif"
        record_match_to_gif(PPO.load(left_path), PPO.load(right_path), filename=gif_path)

    print("Self-play training complete!")

# ---------------------------
# RUN
# ---------------------------
if __name__ == "__main__":
    self_play_training(total_iterations=1, train_timesteps=50000)
