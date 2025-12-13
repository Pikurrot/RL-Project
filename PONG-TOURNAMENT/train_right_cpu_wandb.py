from __future__ import annotations
import os
import socket
HOSTNAME = socket.gethostname()

if HOSTNAME == "cudahpc16":
	# idk who set up this cluster but without this the gpu is not detected
	os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
	os.environ["CUDA_VISIBLE_DEVICES"] = "9"
	os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import numpy as np
import gymnasium as gym
import imageio
from gymnasium.wrappers import AtariPreprocessing, FrameStackObservation
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecMonitor, SubprocVecEnv, DummyVecEnv
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback, CallbackList, BaseCallback
import wandb
from wandb.integration.sb3 import WandbCallback
import ale_py
import torch


RUN_NAME = "pong_right_10M_ent_coef_03"
ROOT_DIR = f"./results/{RUN_NAME}"
LOG_DIR = os.path.join(ROOT_DIR, "logs")
MODEL_DIR = f"/data/users/elopez/checkpoints_pong/{RUN_NAME}"
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

WANDB_ENTITY = "paradigms-team"
WANDB_PROJECT = "PongTournament"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# -------------------------------
# 1. HELPER FUNCTIONS
# -------------------------------

def log_match_video(path: str, label: str, step: int) -> None:
    """Your existing function to log video to WandB."""
    if wandb.run is None:
        return
    # Converts path to string just in case
    wandb.log({f"matches/{label}": wandb.Video(str(path), fps=10, format="gif")},step=step)

def record_match_vs_cpu(model, env_fn, filename="match.gif", max_steps=3000):
    """
    Adapted from your 'record_match' to work with Standard Gym (Agent vs CPU).
    """
    # Create a fresh env for recording with RGB rendering
    env = env_fn()
    
    # Reset
    obs, _ = env.reset()
    frames = []
    done = False
    step = 0

    while not done and step < max_steps:
        # 1. Capture Frame (We need the original RGB frame, not the stacked grayscale obs)
        # Note: In Gym, render() returns the frame if render_mode is set
        frame = env.render() 
        frames.append(frame)

        # 2. Predict Action
        action, _ = model.predict(obs, deterministic=True)
        
        # 3. Step
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        step += 1

    env.close()
    
    # Save using imageio as requested
    imageio.mimsave(filename, frames, fps=30)
    return filename

# -------------------------------
# 2. CUSTOM CALLBACK
# -------------------------------
class VideoLogCallback(BaseCallback):
    def __init__(self, env_fn, freq=100_000, verbose=1):
        super().__init__(verbose)
        self.env_fn = env_fn
        self.freq = freq

    def _on_step(self) -> bool:
        if self.n_calls % self.freq == 0:
            # Define filename
            tmp_path = os.path.join(MODEL_DIR, f"match_step_{self.num_timesteps}.gif")
            
            # 1. Record the match
            if self.verbose > 0:
                print(f"Recording video at step {self.num_timesteps}...")
            
            record_match_vs_cpu(
                model=self.model,
                env_fn=self.env_fn,
                filename=tmp_path
            )
            
            # 2. Log to WandB using your function
            log_match_video(tmp_path, label="agent_vs_cpu", step=self.num_timesteps)
            
        return True

# -------------------------------
# 3. SETUP & PATHS
# -------------------------------

def make_cpu_env():
    # IMPORTANT: render_mode="rgb_array" is required for imageio to capture frames
    env = gym.make("PongNoFrameskip-v4", render_mode="rgb_array")
    env = AtariPreprocessing(env, screen_size=84, grayscale_obs=True, scale_obs=True)
    env = FrameStackObservation(env, stack_size=4)
    return env

if __name__ == "__main__":
    gym.register_envs(ale_py)

    wandb.init(
    entity=WANDB_ENTITY,
    project=WANDB_PROJECT,
    name=RUN_NAME,
    sync_tensorboard=True
    )
    
    # -------------------------------
    # 4. TRAINING SETUP
    # -------------------------------
    n_envs = 8
    env = SubprocVecEnv([make_cpu_env for _ in range(n_envs)])
    env = VecMonitor(env)

    eval_env = DummyVecEnv([make_cpu_env])
    eval_env = VecMonitor(eval_env)

    # -------------------------------
    # 5. CALLBACKS
    # -------------------------------
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=MODEL_DIR,
        log_path=LOG_DIR,
        eval_freq=50_000,
        n_eval_episodes=5,
        deterministic=True,
        render=False
    )
    
    checkpoint_callback = CheckpointCallback(
        save_freq=100_000,
        save_path=os.path.join(MODEL_DIR, "checkpoints"),
        name_prefix="pong_cpu"
    )

    wandb_callback = WandbCallback(
        model_save_path=os.path.join(MODEL_DIR, "wandb_models"),
        verbose=1,
    )
    
    # --- HERE IS YOUR CUSTOM VIDEO CALLBACK ---
    video_callback = VideoLogCallback(
        env_fn=make_cpu_env, 
        freq=100_000  # Record a video every 100k steps
    )

    callback_list = CallbackList([
        eval_callback, 
        checkpoint_callback, 
        wandb_callback, 
        video_callback
    ])

    # -------------------------------
    # 6. RUN MODEL
    # -------------------------------
    model = PPO(
        "CnnPolicy",
        env,
        verbose=1,
        batch_size=256,
        learning_rate=2.5e-4,
        n_steps=128,
        n_epochs=4,
        clip_range=0.1,
		gamma=0.99,
        ent_coef=0.3,
        vf_coef=0.5,
        device=DEVICE,
        tensorboard_log=os.path.join(ROOT_DIR, "tensorboard"),
        policy_kwargs={"normalize_images": False}
    )

    model.learn(
		total_timesteps=10_000_000, 
		callback=callback_list, 
		progress_bar=True,
		log_interval=25
	)
    
    model.save(os.path.join(ROOT_DIR, "model.zip"))
    env.close()
    wandb.finish()
