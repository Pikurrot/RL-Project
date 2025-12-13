from __future__ import annotations
import os
import socket
HOSTNAME = socket.gethostname()

if HOSTNAME == "cudahpc16":
    # idk who set up this cluster but without this the gpu is not detected
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "7"
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import numpy as np
import random
import supersuit as ss
from pettingzoo.atari import pong_v3
import gymnasium as gym
import torch
from gymnasium import spaces
import imageio
from stable_baselines3 import PPO
import stable_baselines3
from stable_baselines3.common.vec_env import DummyVecEnv
import wandb
from wandb.integration.sb3 import WandbCallback
from pathlib import Path

MAX_INT = int(10e6)
RUN_NAME = "pong_both_sides"
N_ENVS = 8
RIGHT_PRETRAINED_PATH = Path("/data/users/elopez/checkpoints_pong/pong_right_10M_ent_coef_001/best_model.zip")
CHECKPOINT_DIR = Path(f"/data/users/elopez/checkpoints_pong/{RUN_NAME}")
VIDEOS_DIR = Path(f"./videos/{RUN_NAME}")
WANDB_ENTITY = "paradigms-team"
WANDB_PROJECT = "PongTournament"
WANDB_RUN_NAME = RUN_NAME
SCRIPT_NAME = Path(__file__).stem
CYCLES = 10
CYCLE_TIMESTEPS = 100_000

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
        """Returns the opponent action using a PPO-style model."""
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

def policy_predict(model, obs, deterministic=True):
    """Returns an action for PPO opponents."""
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

def flip_model_file(src_path: Path, dst_path: Path) -> Path:
    """
    Loads a PPO model, mirrors its convolutional filters and the first linear
    layer (NatureCNN layout), saves to dst_path, and returns dst_path.
    This mirrors the behavior from flip_model.py so a right-paddle agent
    can be turned into a left-paddle agent.
    """
    print(f"Flipping model from {src_path} -> {dst_path}")
    model = PPO.load(src_path, device="cpu")
    policy = model.policy

    with torch.no_grad():
        conv_layers = policy.features_extractor.cnn
        conv_layers[0].weight.data = conv_layers[0].weight.data.flip(-1)
        conv_layers[2].weight.data = conv_layers[2].weight.data.flip(-1)
        conv_layers[4].weight.data = conv_layers[4].weight.data.flip(-1)

        linear_layer = policy.features_extractor.linear[0]
        weights = linear_layer.weight.data

        n_channels, h, w = 64, 7, 7
        expected_size = n_channels * h * w
        if weights.shape[1] == expected_size:
            reshaped_weights = weights.view(weights.shape[0], n_channels, h, w)
            flipped_weights = reshaped_weights.flip(3)
            linear_layer.weight.data = flipped_weights.flatten(start_dim=1)
        else:
            print(
                f"WARNING: Unexpected feature size {weights.shape[1]} "
                f"(expected {expected_size}). Skipping linear flip."
            )

    model.save(dst_path)
    return dst_path

right_model = PPO.load(RIGHT_PRETRAINED_PATH, device=DEVICE)

# Build initial left model by flipping the right model
initial_left_path = CHECKPOINT_DIR / "left_iter_0.zip"
flip_model_file(RIGHT_PRETRAINED_PATH, initial_left_path)
left_model = PPO.load(initial_left_path, device=DEVICE)

print(f"=== PRE-TRAIN: GENERANDO GIF ===")
gif_path = record_match(left_model, right_model, filename=VIDEOS_DIR / "match_iter_0.gif")
log_match_video(Path(gif_path), "iter_0")

for i in range(CYCLES):
    print(f"=== CICLO {i}: ENTRENANDO RIGHT ===")
    
    current_right_env = right_model.get_env()
    if current_right_env is not None:
        current_right_env.close()
    env_right = make_vector_env("first_0", opponent_model=left_model)
    right_model.set_env(env_right)
    right_model.learn(
        CYCLE_TIMESTEPS,
        log_interval=30,
        reset_num_timesteps=False,
		progress_bar=True,
        callback=make_wandb_callback(f"right_iter_{i}"),
    )
    right_cycle_avg_reward = evaluate_agent(
        agent=right_model,
        opponent=left_model,
        player_id="first_0",
    )
    log_average_reward("right", right_cycle_avg_reward, step=right_model.num_timesteps)
    right_checkpoint = CHECKPOINT_DIR / f"right_iter_{i}.zip"
    right_model.save(right_checkpoint)

    # Update left_model by flipping the freshly trained right model
    flipped_left_path = CHECKPOINT_DIR / f"left_iter_{i+1}.zip"
    flip_model_file(right_checkpoint, flipped_left_path)
    left_model = PPO.load(flipped_left_path, device=DEVICE)

    print(f"=== CICLO {i}: GENERANDO GIF ===")
    gif_path = record_match(left_model, right_model, filename=VIDEOS_DIR / f"match_iter_{i+1}.gif")
    log_match_video(Path(gif_path), f"iter_{i+1}")

wandb.finish()
