from __future__ import annotations
import os
import socket
HOSTNAME = socket.gethostname()

if HOSTNAME == "cudahpc16":
    # idk who set up this cluster but without this the gpu is not detected
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "5"
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import numpy as np
import random
import supersuit as ss
from pettingzoo.atari import pong_v3
import gymnasium as gym
import torch
from gymnasium import spaces
import ale_py
import torch.nn as nn
import torch.nn.functional as F
import imageio
from stable_baselines3 import PPO
import stable_baselines3
import wandb
from wandb.integration.sb3 import WandbCallback
from pathlib import Path

gym.register_envs(ale_py)

MAX_INT = int(10e6)
TIME_STEP_MAX = 100000
CHECKPOINT_DIR = Path("/data/users/elopez/checkpoints_pong")
VIDEOS_DIR = Path("./videos")
WANDB_ENTITY = "paradigms-team"
WANDB_PROJECT = "PongTorunament"
WANDB_RUN_NAME = "pong_v1"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
VIDEOS_DIR.mkdir(parents=True, exist_ok=True)
(CHECKPOINT_DIR / "wandb_models").mkdir(parents=True, exist_ok=True)

wandb_run = wandb.init(
    entity=WANDB_ENTITY,
    project=WANDB_PROJECT,
    name=WANDB_RUN_NAME,
    config={
        "total_timesteps_initial": 300_000,
        "total_timesteps_cycle": 100_000,
        "time_step_max": TIME_STEP_MAX,
        "device": DEVICE.type,
    },
)

def get_seed(MAX_INT=int(10e6)):
    return random.randint(0, MAX_INT)

def make_env(render_mode="rgb_array"):
    env = pong_v3.env(num_players=2, render_mode=render_mode)

    # env = ss.sticky_actions_v0(env, repeat_action_probability=0.25)
    env = ss.color_reduction_v0(env, mode="B")
    env = ss.resize_v1(env, x_size=84, y_size=84)
    env = ss.frame_stack_v1(env, 4, stack_dim=0)
    env = ss.dtype_v0(env, dtype=np.float32)
    env = ss.normalize_obs_v0(env, env_min=0, env_max=1)
    env = ss.reshape_v0(env, (4, 84, 84))

    env.reset(seed=get_seed(MAX_INT))
    env.action_space(env.possible_agents[0]).seed(get_seed(MAX_INT))

    return env

class PZSingleAgentWrapper(gym.Env):
    metadata = {"render_modes": ["rgb_array", "human"]}

    def __init__(self, player_id, opponent_model, render_mode=None):
        super().__init__()

        self.player_id = player_id
        self.opponent_id = "second_0" if player_id == "first_0" else "first_0"

        # Inicializa el entorno
        self.env = make_env(render_mode=render_mode)

        # Observación de ejemplo
        example_obs = self.env.observe(self.player_id)

        # Space debe ser uint8 para CnnPolicy
        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=example_obs.shape,
            dtype=np.uint8
        )

        # Espacio de acciones del agente
        self.action_space = self.env.action_space(self.player_id)

        self.opponent_model = opponent_model

        self.agent_iter = None  # Guardaremos el iterador

    def reset(self, *, seed=None, options=None):
        self.env.reset(seed=seed)
        self.agent_iter = iter(self.env.agent_iter())
    
        obs_agent = None
        while True:
            agent = next(self.agent_iter)
            obs, reward, terminated, truncated, info = self.env.last()
    
            if agent == self.player_id:
                obs_agent = obs
                self.env.step(0)
                break
    
            elif isinstance(self.opponent_model, DQN):
                with torch.no_grad():
                    t = torch.tensor(obs, dtype=torch.float32, device=DEVICE).unsqueeze(0) / 255.0
                    act = self.opponent_model(t).argmax().item()

            else:
                act, _ = self.opponent_model.predict(obs, deterministic=True)

            self.env.step(act)

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
                self.env.step(0)
                break
    
            elif isinstance(self.opponent_model, DQN):
                with torch.no_grad():
                    t = torch.tensor(obs, dtype=torch.float32, device=DEVICE).unsqueeze(0) / 255.0
                    act = self.opponent_model(t).argmax().item()

            else:
                act, _ = self.opponent_model.predict(obs, deterministic=True)

            self.env.step(act)

        return obs_agent, reward_agent, done, False, {}

    def render(self):
        return self.env.render()

    def close(self):
        self.env.close()

class DQN(nn.Module):
    def __init__(self, in_channels, n_actions):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.fc1 = nn.Linear(64 * 7 * 7, 512)
        self.out = nn.Linear(512, n_actions)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.out(x)

opponent_model = DQN(4, 6).to(DEVICE)
opponent_model.load_state_dict(
    torch.load(CHECKPOINT_DIR / "checkpoint_best.pth", map_location=DEVICE)["policy_state_dict"]
)

def make_wandb_callback(label: str) -> WandbCallback:
    return WandbCallback(
        model_save_path=str(CHECKPOINT_DIR / "wandb_models" / label),
        model_save_freq=0,
        gradient_save_freq=0,
        verbose=1,
    )

def log_match_video(path: Path, label: str) -> None:
    if wandb.run is None:
        return
    wandb.log({f"matches/{label}": wandb.Video(str(path), fps=30, format="gif")})

def record_match(left_model, right_model, filename="match.gif", max_steps=3000):
    """
    Simula un partido entre left_model y right_model y lo guarda como GIF.
    """
    env = make_env(render_mode="rgb_array")
    env.reset()

    agent_iter = iter(env.agent_iter())

    frames = []

    # Inicializar observaciones
    obs_left = env.observe("second_0")
    obs_right = env.observe("first_0")

    done = False
    step = 0

    while not done and step < max_steps:

        # Acciones para cada agente
        with torch.no_grad():
            act_left = left_model.predict(obs_left, deterministic=True)[0]
            act_right = right_model.predict(obs_right, deterministic=True)[0]

        # SOLO un paso del agente actual
        agent = next(agent_iter)
        obs, reward, termination, truncation, info = env.last()

        if agent == "second_0":
            env.step(act_left)
            obs_left = obs

        elif agent == "first_0":
            env.step(act_right)
            obs_right = obs

        else:
            env.step(env.action_space(agent).sample())

        done = termination or truncation

        # Capturar frame
        frame = env.render()
        frames.append(frame)

        step += 1

    env.close()
    imageio.mimsave(filename, frames, fps=30)
    return filename


env_left = PZSingleAgentWrapper(player_id="second_0", opponent_model=opponent_model)

print("LEFT vs DQN")

model_left = PPO("CnnPolicy", env_left, verbose=1, device=DEVICE)
model_left.learn(
    total_timesteps=300_000,
    progress_bar=True,
    log_interval=25,
    callback=make_wandb_callback("left_initial"),
)
model_left.save(CHECKPOINT_DIR / "left1.zip")

left_frozen = PPO.load(CHECKPOINT_DIR / "left1.zip", device=DEVICE)

env_right = PZSingleAgentWrapper(player_id="first_0", opponent_model=left_frozen)

print("RIGHT vs LEFT")
model_right = PPO("CnnPolicy", env_right, verbose=1, device=DEVICE)
model_right.learn(
    total_timesteps=300_000,
    progress_bar=True,
    log_interval=25,
    callback=make_wandb_callback("right_initial"),
)
model_right.save(CHECKPOINT_DIR / "right1.zip")


left_model = PPO.load(CHECKPOINT_DIR / "left1.zip", device=DEVICE)
right_model = PPO.load(CHECKPOINT_DIR / "right1.zip", device=DEVICE)

print(f"=== PRE-TRAIN: GENERANDO GIF ===")
gif_path = record_match(left_model, right_model, filename=VIDEOS_DIR / "match_iter_0.gif")
log_match_video(Path(gif_path), "iter_0")

for i in range(20):
    print(f"=== CICLO {i}: ENTRENANDO LEFT ===")
    
    env_left = PZSingleAgentWrapper("second_0", opponent_model=right_model)
    left_model.set_env(env_left)
    left_model.learn(
        200000,
        log_interval=30,
        reset_num_timesteps=False,
        callback=make_wandb_callback(f"left_iter_{i}"),
    )
    left_model.save(CHECKPOINT_DIR / f"left_iter_{i}.zip")

    print(f"=== CICLO {i}: ENTRENANDO RIGHT ===")
    
    env_right = PZSingleAgentWrapper("first_0", opponent_model=left_model)
    right_model.set_env(env_right)
    right_model.learn(
        200000,
        log_interval=30,
        reset_num_timesteps=False,
        callback=make_wandb_callback(f"right_iter_{i}"),
    )
    right_model.save(CHECKPOINT_DIR / f"right_iter_{i}.zip")

    print(f"=== CICLO {i}: GENERANDO GIF ===")
    gif_path = record_match(left_model, right_model, filename=VIDEOS_DIR / f"match_iter_{i+1}.gif")
    log_match_video(Path(gif_path), f"iter_{i+1}")

wandb.finish()
