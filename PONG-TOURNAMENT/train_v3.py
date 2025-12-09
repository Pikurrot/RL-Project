from __future__ import annotations
import os
import socket
HOSTNAME = socket.gethostname()

if HOSTNAME == "cudahpc16":
	# idk who set up this cluster but without this the gpu is not detected
	os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
	os.environ["CUDA_VISIBLE_DEVICES"] = "8"
	os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import numpy as np
import random
import supersuit as ss
from pettingzoo.atari import pong_v3
import gymnasium as gym
import torch
from gymnasium import spaces
import torch.nn as nn
import torch.nn.functional as F
import imageio
from stable_baselines3 import PPO
import stable_baselines3
from stable_baselines3.common.base_class import BaseAlgorithm
import wandb
from wandb.integration.sb3 import WandbCallback
from pathlib import Path
from typing import Callable, List, Tuple

MAX_INT = int(10e6)
TIME_STEP_MAX = 100000
CHECKPOINT_DIR = Path("/data/users/elopez/checkpoints_pong3")
VIDEOS_DIR = Path("./videos3")
WANDB_ENTITY = "paradigms-team"
WANDB_PROJECT = "PongTorunament"
WANDB_RUN_NAME = "pong_v3"
SCRIPT_NAME = Path(__file__).stem

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OPPONENT_EXPLORATION_PROB = 0.05

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


class OpponentPool:
    """Stores opponent factories so we can sample past versions."""

    def __init__(self, name: str):
        self.name = name
        self._entries: List[Tuple[str, Callable[[], object]]] = []
        self._rng = random.Random()

    def add(self, label: str, factory: Callable[[], object]) -> None:
        self._entries.append((label, factory))

    def sample_model(self):
        if not self._entries:
            raise RuntimeError(f"Opponent pool '{self.name}' is empty.")
        label, factory = self._rng.choice(self._entries)
        return factory()

    def __len__(self) -> int:
        return len(self._entries)


def register_ppo_checkpoint(pool: OpponentPool, label: str, checkpoint_path: Path) -> None:
    """Adds a PPO checkpoint loader to the given opponent pool."""
    pool.add(
        label,
        lambda path=str(checkpoint_path): PPO.load(path, device=DEVICE),
    )

def get_seed(MAX_INT=int(10e6)):
    return random.randint(0, MAX_INT)

def make_env(render_mode="rgb_array"):
    env = pong_v3.env(num_players=2, render_mode=render_mode)

    env = ss.sticky_actions_v0(env, repeat_action_probability=0.25)
    env = ss.color_reduction_v0(env, mode="B")
    env = ss.resize_v1(env, x_size=84, y_size=84)
    env = ss.frame_stack_v1(env, 4, stack_dim=0)
    env = ss.dtype_v0(env, dtype=np.uint8)
    env = ss.reshape_v0(env, (4, 84, 84))

    env.reset(seed=get_seed(MAX_INT))
    env.action_space(env.possible_agents[0]).seed(get_seed(MAX_INT))

    return env

def _is_model_provider(candidate) -> bool:
    """Returns True if candidate is a zero-arg factory, not a policy instance."""
    return callable(candidate) and not isinstance(candidate, (nn.Module, BaseAlgorithm))


class PZSingleAgentWrapper(gym.Env):
    metadata = {"render_modes": ["rgb_array", "human"]}

    def __init__(
        self,
        player_id,
        opponent_model,
        render_mode=None,
        opponent_deterministic=False,
        opponent_exploration=OPPONENT_EXPLORATION_PROB,
    ):
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
        self.opponent_action_space = self.env.action_space(self.opponent_id)

        if _is_model_provider(opponent_model):
            self._opponent_provider = opponent_model
        else:
            self._opponent_provider = lambda: opponent_model

        self.opponent_model = self._opponent_provider()
        self.opponent_deterministic = opponent_deterministic
        self.opponent_exploration = opponent_exploration

        self.agent_iter = None  # Guardaremos el iterador

    def _refresh_opponent_model(self) -> None:
        self.opponent_model = self._opponent_provider()

    def _should_explore(self) -> bool:
        return self.opponent_exploration > 0 and random.random() < self.opponent_exploration
    
    def _opponent_act(self, obs):
        """Returns the opponent action depending on model type."""
        if self._should_explore():
            return self.opponent_action_space.sample()

        if isinstance(self.opponent_model, DQN):
            with torch.no_grad():
                t = torch.tensor(obs, dtype=torch.float32, device=DEVICE).unsqueeze(0) / 255.0
                return self.opponent_model(t).argmax().item()
        else:
            act, _ = self.opponent_model.predict(
                obs,
                deterministic=self.opponent_deterministic,
            )
            return act

    def reset(self, *, seed=None, options=None):
        self._refresh_opponent_model()
        self.env.reset(seed=seed)
        self.agent_iter = iter(self.env.agent_iter())
    
        obs_agent = None
        while True:
            agent = next(self.agent_iter)
            obs, reward, terminated, truncated, info = self.env.last()
    
            if agent == self.player_id:
                return obs, {}

            if terminated or truncated:
                self.env.step(None)
                continue

            act = self._opponent_act(obs)
            self.env.step(act)

    def step(self, action):
        obs_agent, reward_agent, done = None, 0, False

        # Player acts first (env is waiting for this action after reset/last loop)
        self.env.step(action)

        while True:
            agent = next(self.agent_iter)
            obs, reward, terminated, truncated, info = self.env.last()
    
            if agent == self.player_id:
                obs_agent = obs
                reward_agent = reward
                done = terminated or truncated
                return obs_agent, reward_agent, done, False, {}

            if terminated or truncated:
                self.env.step(None)
                continue

            act = self._opponent_act(obs)
            self.env.step(act)

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
opponent_model.eval()

right_opponent_pool = OpponentPool("right_players")
right_opponent_pool.add("pretrained_dqn", lambda: opponent_model)
left_opponent_pool = OpponentPool("left_players")

def policy_predict(model, obs, deterministic=True):
    """Returns an action for either PPO or DQN opponents."""
    if isinstance(model, DQN):
        with torch.no_grad():
            t = (
                torch.tensor(obs, dtype=torch.float32, device=DEVICE)
                .unsqueeze(0)
                .div_(255.0)
            )
            return int(model(t).argmax(dim=1).item())

    action, _ = model.predict(obs, deterministic=deterministic)
    if isinstance(action, np.ndarray):
        if action.ndim == 0:
            return int(action.item())
        if action.size == 1:
            return int(action.flatten()[0])
    return action

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

def build_video_path(tag: str) -> Path:
    return VIDEOS_DIR / f"{SCRIPT_NAME}_{tag}.gif"

def evaluate_agent(agent, opponent, player_id: str, episodes: int = 5) -> float:
    """Runs short rollouts to estimate the agent reward versus a fixed opponent."""
    env = PZSingleAgentWrapper(
        player_id=player_id,
        opponent_model=opponent,
        opponent_deterministic=True,
        opponent_exploration=0.0,
    )
    rewards = []
    for _ in range(episodes):
        obs, _ = env.reset()
        done = False
        ep_reward = 0.0
        while not done:
            action = policy_predict(agent, obs, deterministic=True)
            obs, reward, done, _, _ = env.step(action)
            ep_reward += reward
        rewards.append(ep_reward)
    env.close()
    if not rewards:
        return 0.0
    return float(np.mean(rewards))

def log_average_reward(agent_label: str, avg_reward: float, step: int | None = None) -> None:
    if wandb.run is None:
        return
    payload = {f"rewards/{agent_label}": avg_reward}
    if step is None:
        wandb.log(payload)
    else:
        wandb.log(payload, step=step)

def record_match(left_model, right_model, filename="match.gif", max_steps=3000):
    """
    Simula un partido entre left_model y right_model y lo guarda como GIF.
    """
    env = make_env(render_mode="rgb_array")
    env.reset()

    agent_iter = iter(env.agent_iter())

    frames = []

    done = False
    step = 0

    while not done and step < max_steps:
        agent = next(agent_iter)
        obs, reward, termination, truncation, info = env.last()

        if agent == "second_0":
            act_left = policy_predict(left_model, obs, deterministic=True)
            env.step(act_left)

        elif agent == "first_0":
            act_right = policy_predict(right_model, obs, deterministic=True)
            env.step(act_right)

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


env_left = PZSingleAgentWrapper(
    player_id="second_0",
    opponent_model=right_opponent_pool.sample_model,
    opponent_exploration=OPPONENT_EXPLORATION_PROB,
)

print("LEFT vs DQN")

model_left = PPO("CnnPolicy", env_left, verbose=1, device=DEVICE)
pre_left_gif = record_match(
    model_left,
    opponent_model,
    filename=build_video_path("match_pre_left"),
)
log_match_video(Path(pre_left_gif), "pre_left")
model_left.learn(
    total_timesteps=300_000,
    progress_bar=True,
    log_interval=25,
    callback=make_wandb_callback("left_initial"),
)
post_left_gif = record_match(
    model_left,
    opponent_model,
    filename=build_video_path("match_post_left"),
)
log_match_video(Path(post_left_gif), "post_left")
left_initial_avg_reward = evaluate_agent(
    agent=model_left,
    opponent=opponent_model,
    player_id="second_0",
)
log_average_reward("left", left_initial_avg_reward, step=model_left.num_timesteps)
left_initial_ckpt = CHECKPOINT_DIR / "left1.zip"
model_left.save(left_initial_ckpt)
register_ppo_checkpoint(left_opponent_pool, "left_initial_ckpt", left_initial_ckpt)
left_model = PPO.load(left_initial_ckpt, device=DEVICE)
left_opponent_pool.add("left_live", lambda: left_model)

env_right = PZSingleAgentWrapper(
    player_id="first_0",
    opponent_model=left_opponent_pool.sample_model,
    opponent_exploration=OPPONENT_EXPLORATION_PROB,
)

print("RIGHT vs LEFT")
model_right = PPO("CnnPolicy", env_right, verbose=1, device=DEVICE)
model_right.learn(
    total_timesteps=300_000,
    progress_bar=True,
    log_interval=25,
    callback=make_wandb_callback("right_initial"),
)
right_initial_avg_reward = evaluate_agent(
    agent=model_right,
    opponent=left_model,
    player_id="first_0",
)
log_average_reward("right", right_initial_avg_reward, step=model_right.num_timesteps)
post_right_gif = record_match(
    left_model=left_model,
    right_model=model_right,
    filename=build_video_path("match_post_right"),
)
log_match_video(Path(post_right_gif), "post_right")
right_initial_ckpt = CHECKPOINT_DIR / "right1.zip"
model_right.save(right_initial_ckpt)
register_ppo_checkpoint(right_opponent_pool, "right_initial_ckpt", right_initial_ckpt)
right_model = PPO.load(right_initial_ckpt, device=DEVICE)
right_opponent_pool.add("right_live", lambda: right_model)

print(f"=== PRE-TRAIN: GENERANDO GIF ===")
gif_path = record_match(left_model, right_model, filename=VIDEOS_DIR / "match_iter_0.gif")
log_match_video(Path(gif_path), "iter_0")

for i in range(10):
    print(f"=== CICLO {i}: ENTRENANDO LEFT ===")
    
    env_left = PZSingleAgentWrapper(
        "second_0",
        opponent_model=right_opponent_pool.sample_model,
        opponent_exploration=OPPONENT_EXPLORATION_PROB,
    )
    left_model.set_env(env_left)
    left_model.learn(
        100000,
        log_interval=30,
        reset_num_timesteps=False,
        callback=make_wandb_callback(f"left_iter_{i}"),
    )
    left_cycle_avg_reward = evaluate_agent(
        agent=left_model,
        opponent=right_model,
        player_id="second_0",
    )
    log_average_reward("left", left_cycle_avg_reward, step=left_model.num_timesteps)
    left_iter_ckpt = CHECKPOINT_DIR / f"left_iter_{i}.zip"
    left_model.save(left_iter_ckpt)
    register_ppo_checkpoint(left_opponent_pool, f"left_iter_{i}", left_iter_ckpt)

    print(f"=== CICLO {i}: ENTRENANDO RIGHT ===")
    
    env_right = PZSingleAgentWrapper(
        "first_0",
        opponent_model=left_opponent_pool.sample_model,
        opponent_exploration=OPPONENT_EXPLORATION_PROB,
    )
    right_model.set_env(env_right)
    right_model.learn(
        100000,
        log_interval=30,
        reset_num_timesteps=False,
        callback=make_wandb_callback(f"right_iter_{i}"),
    )
    right_cycle_avg_reward = evaluate_agent(
        agent=right_model,
        opponent=left_model,
        player_id="first_0",
    )
    log_average_reward("right", right_cycle_avg_reward, step=right_model.num_timesteps)
    right_iter_ckpt = CHECKPOINT_DIR / f"right_iter_{i}.zip"
    right_model.save(right_iter_ckpt)
    register_ppo_checkpoint(right_opponent_pool, f"right_iter_{i}", right_iter_ckpt)

    print(f"=== CICLO {i}: GENERANDO GIF ===")
    gif_path = record_match(left_model, right_model, filename=VIDEOS_DIR / f"match_iter_{i+1}.gif")
    log_match_video(Path(gif_path), f"iter_{i+1}")

wandb.finish()
