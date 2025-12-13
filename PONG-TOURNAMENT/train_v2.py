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
import torch.nn as nn
import torch.nn.functional as F
import imageio
from stable_baselines3 import PPO
import stable_baselines3
from stable_baselines3.common.vec_env import DummyVecEnv
import wandb
from wandb.integration.sb3 import WandbCallback
from pathlib import Path

MAX_INT = int(10e6)
RUN_NAME = "distance_penalty_float_solved"
N_ENVS = 16
DQN_CHECKPOINT = Path("/data/users/elopez/checkpoints_pong/checkpoint_best.pth")
CHECKPOINT_DIR = Path(f"/data/users/elopez/checkpoints_pong/{RUN_NAME}")
VIDEOS_DIR = Path(f"./videos/{RUN_NAME}")
WANDB_ENTITY = "paradigms-team"
WANDB_PROJECT = "PongTorunament"
WANDB_RUN_NAME = RUN_NAME
SCRIPT_NAME = Path(__file__).stem

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
VIDEOS_DIR.mkdir(parents=True, exist_ok=True)
(CHECKPOINT_DIR / "wandb_models").mkdir(parents=True, exist_ok=True)

wandb_run = wandb.init(
    entity=WANDB_ENTITY,
    project=WANDB_PROJECT,
    name=WANDB_RUN_NAME
)

def get_seed(MAX_INT=int(10e6)):
    return random.randint(0, MAX_INT)

def make_env(render_mode="rgb_array"):
    env = pong_v3.env(num_players=2, render_mode=render_mode)

    env = ss.color_reduction_v0(env, mode="B")
    env = ss.resize_v1(env, x_size=84, y_size=84)
    env = ss.frame_stack_v1(env, 4, stack_dim=0)
    env = ss.dtype_v0(env, dtype=np.float32)
    env = ss.normalize_obs_v0(env, env_min=0, env_max=1)
    env = ss.reshape_v0(env, (4, 84, 84))

    env.reset(seed=get_seed(MAX_INT))
    env.action_space(env.possible_agents[0]).seed(get_seed(MAX_INT))

    return env


def find_ball(obs):
	y_min, y_max = 14, 78
	obs_crop = obs[y_min:y_max, :]
	obs_crop = obs_crop > 0.5
	# get center of mass
	center_of_mass = np.argwhere(obs_crop)
	center_of_mass = np.mean(center_of_mass, axis=0)
	center_of_mass[0] += y_min
	# if is nan, return None
	if np.isnan(center_of_mass).any():
		return (None, None)
	return center_of_mass # (y, x)

def get_agent_position(obs, player_id):
	assert player_id in ["first_0", "second_0"]
	y_min, y_max = 14, 78
	x_threshold = 40
	if player_id == "second_0": # left player
		obs_crop = obs[y_min:y_max, :x_threshold]
	else: # right player
		obs_crop = obs[y_min:y_max, x_threshold:]
	obs_crop = (obs_crop > 0.1) & (obs_crop < 0.3)
	center_of_mass = np.argwhere(obs_crop)
	center_of_mass = np.mean(center_of_mass, axis=0)
	center_of_mass[0] += y_min
	if player_id == "first_0":
		center_of_mass[1] += x_threshold
	if np.isnan(center_of_mass).any():
		return (None, None)
	return center_of_mass # (y, x)


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
            high=1,
            shape=example_obs.shape,
            dtype=np.float32
        )

        # Espacio de acciones del agente
        self.action_space = self.env.action_space(self.player_id)

        self.opponent_model = opponent_model

        self.agent_iter = None  # Guardaremos el iterador
    
    def _opponent_act(self, obs):
        """Returns the opponent action depending on model type."""
        if isinstance(self.opponent_model, DQN):
            # Observations are already normalized to [0, 1], no further scaling needed
            with torch.no_grad():
                t = torch.tensor(obs, dtype=torch.float32, device=DEVICE).unsqueeze(0)
                return self.opponent_model(t).argmax().item()
        else:
            act, _ = self.opponent_model.predict(obs, deterministic=True)
            if isinstance(act, np.ndarray):
                if act.ndim == 0:
                    act = int(act.item())
                else:
                    act = int(act.flatten()[0])
            return act

    def reset(self, *, seed=None, options=None):
        self.env.reset(seed=seed)
        self.agent_iter = iter(self.env.agent_iter())
    
        obs_agent = None
        while True:
            agent = next(self.agent_iter)
            obs, reward, terminated, truncated, info = self.env.last()
    
            if agent == self.player_id:
                return obs, {}

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
                # Cancel any reward
                reward = 0
                # Penalty for y distance to ball
                ball_y, ball_x = find_ball(obs)
                agent_y, agent_x = get_agent_position(obs, self.player_id)
                if ball_y is not None and agent_y is not None:
                    reward -= abs(ball_y - agent_y) * 0.1
                reward_agent = reward
                done = terminated or truncated
                if done:
                    # Advance the AEC env after termination/truncation
                    self.env.step(None)
                return obs_agent, reward_agent, done, False, {}

            if terminated or truncated:
                self.env.step(None)
            else:
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
    torch.load(DQN_CHECKPOINT, map_location=DEVICE)["policy_state_dict"]
)

def policy_predict(model, obs, deterministic=True):
    """Returns an action for either PPO or DQN opponents."""
    if isinstance(model, DQN):
        with torch.no_grad():
            # Observations already normalized to [0, 1]
            t = torch.tensor(obs, dtype=torch.float32, device=DEVICE).unsqueeze(0)
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

def make_vector_env(player_id: str, opponent_model, n_envs: int = N_ENVS):
    """Creates a simple vectorized env to speed up data collection."""
    return DummyVecEnv(
        [lambda: PZSingleAgentWrapper(player_id=player_id, opponent_model=opponent_model) for _ in range(n_envs)]
    )

def evaluate_agent(agent, opponent, player_id: str, episodes: int = 5) -> float:
    """Runs short rollouts to estimate the agent reward versus a fixed opponent."""
    env = PZSingleAgentWrapper(player_id=player_id, opponent_model=opponent)
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


env_left = make_vector_env(player_id="second_0", opponent_model=opponent_model)

print("LEFT vs DQN")

model_left = PPO(
    "CnnPolicy",
    env_left,
    verbose=1,
    device=DEVICE,
    policy_kwargs={"normalize_images": False},
)
pre_left_gif = record_match(
    model_left,
    opponent_model,
    filename=build_video_path("match_pre_left"),
)
log_match_video(Path(pre_left_gif), "pre_left")
model_left.learn(
    total_timesteps=100_000,
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
model_left.save(CHECKPOINT_DIR / "left1.zip")

left_frozen = PPO.load(CHECKPOINT_DIR / "left1.zip", device=DEVICE)

env_right = make_vector_env(player_id="first_0", opponent_model=left_frozen)

print("RIGHT vs LEFT")
model_right = PPO(
    "CnnPolicy",
    env_right,
    verbose=1,
    device=DEVICE,
    policy_kwargs={"normalize_images": False},
)
model_right.learn(
    total_timesteps=100_000,
    progress_bar=True,
    log_interval=25,
    callback=make_wandb_callback("right_initial"),
)
right_initial_avg_reward = evaluate_agent(
    agent=model_right,
    opponent=model_left,
    player_id="first_0",
)
log_average_reward("right", right_initial_avg_reward, step=model_right.num_timesteps)
post_right_gif = record_match(
    left_model=model_left,
    right_model=model_right,
    filename=build_video_path("match_post_right"),
)
log_match_video(Path(post_right_gif), "post_right")
model_right.save(CHECKPOINT_DIR / "right1.zip")


left_model = PPO.load(CHECKPOINT_DIR / "left1.zip", device=DEVICE)
right_model = PPO.load(CHECKPOINT_DIR / "right1.zip", device=DEVICE)

print(f"=== PRE-TRAIN: GENERANDO GIF ===")
gif_path = record_match(left_model, right_model, filename=VIDEOS_DIR / "match_iter_0.gif")
log_match_video(Path(gif_path), "iter_0")

for i in range(10):
    print(f"=== CICLO {i}: ENTRENANDO LEFT ===")
    
    current_left_env = left_model.get_env()
    if current_left_env is not None:
        current_left_env.close()
    env_left = make_vector_env("second_0", opponent_model=right_model)
    left_model.set_env(env_left)
    left_model.learn(
        50000,
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
    left_model.save(CHECKPOINT_DIR / f"left_iter_{i}.zip")

    print(f"=== CICLO {i}: ENTRENANDO RIGHT ===")
    
    current_right_env = right_model.get_env()
    if current_right_env is not None:
        current_right_env.close()
    env_right = make_vector_env("first_0", opponent_model=left_model)
    right_model.set_env(env_right)
    right_model.learn(
        50000,
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
    right_model.save(CHECKPOINT_DIR / f"right_iter_{i}.zip")

    print(f"=== CICLO {i}: GENERANDO GIF ===")
    gif_path = record_match(left_model, right_model, filename=VIDEOS_DIR / f"match_iter_{i+1}.gif")
    log_match_video(Path(gif_path), f"iter_{i+1}")

wandb.finish()
