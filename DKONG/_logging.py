from __future__ import annotations
import wandb
import yaml
import logging
import numpy as np
from pathlib import Path
from wandb.integration.sb3 import WandbCallback as SB3WandbCallback
from stable_baselines3.common.callbacks import EvalCallback, CallbackList, BaseCallback

from environment.env import make_eval_vec_env


DKONG_ROOT = Path(__file__).resolve().parents[0]
CONFIG_PATH = DKONG_ROOT / "config.yml"
PRETRAIN_CONFIG_PATH = DKONG_ROOT / "config_pretrain.yml"
logger = logging.getLogger(__name__)

def load_config() -> dict:
	with open(CONFIG_PATH, "r") as f:
		return yaml.safe_load(f)

def load_pretrain_config() -> dict:
	with open(PRETRAIN_CONFIG_PATH, "r") as f:
		return yaml.safe_load(f)

def init_wandb(config: dict) -> wandb.Run:
	# remove project and run_name from config
	wandb_config = config["wandb"].copy()
	entity = wandb_config.pop("entity")
	project = wandb_config.pop("project")
	run_name = wandb_config.pop("run_name")
	return wandb.init(
		entity=entity,
		project=project,
		name=run_name,
		config=wandb_config,
		sync_tensorboard=True,
		monitor_gym=True,
		save_code=True,
	)

class WandbCallback(SB3WandbCallback):
	def __init__(self, *args, config: dict | None = None, **kwargs):
		super().__init__(*args, **kwargs)
		self.config = config

	def save_model(self) -> None:
		# same as super, but without logging model to wandb (i have limited disk there)
		if self.verbose > 1:
			logger.info(f"Saving model checkpoint to {self.path}")
		# check if path exists, and if so, change to path_i
		if self.path is not None:
			path = Path(self.path)
			save_name = "model"
			i = 1
			while path.exists():
				path = path.parent / f"{save_name}_{i}.zip"
				i += 1
			path = str(path)
			self.model.save(path)


# Called after every evaluation to reset the video recorder
class EvalVideoControlCallback(BaseCallback):
	def __init__(self, eval_env, video_every: int):
		super().__init__()
		self.eval_env = eval_env
		self.video_every = max(1, int(video_every))
		self.completed_evals = 0

	def _on_step(self) -> bool:
		self.completed_evals += 1
		# Reset recorder counter so next evaluation can start a fresh capture
		if hasattr(self.eval_env, "step_id"):
			self.eval_env.step_id = 0
		record_flag = getattr(self.eval_env, "_record_flag", None)
		if record_flag is not None:
			next_eval_idx = self.completed_evals + 1
			should_record_next = (
				self.video_every == 1 or next_eval_idx % self.video_every == 0
			)
			record_flag["record"] = should_record_next
		return True


class LifeStatsAggregator:
	def __init__(self, n_levels: int = 7):
		self.n_levels = n_levels
		self.reset_state(1)

	def reset_state(self, n_envs: int):
		self.n_envs = n_envs
		self.current_max = np.zeros(n_envs, dtype=np.int64)
		self.level_counts = np.zeros(self.n_levels, dtype=np.int64)
		self.screen_completions = 0
		self.completed_lives = 0

	def process_step(self, infos, dones):
		for idx in range(self.n_envs):
			info = infos[idx] if infos is not None and idx < len(infos) else {}
			level = info.get("mario_level", None)
			if level is not None and not np.isnan(level):
				level = int(np.clip(level, 0, self.n_levels - 1))
				self.current_max[idx] = max(self.current_max[idx], level)

			if info.get("screen_completed", False):
				self.screen_completions += 1

			life_end = info.get("life_lost", False) or (dones is not None and idx < len(dones) and dones[idx])
			if life_end:
				self.completed_lives += 1
				max_level = int(self.current_max[idx])
				self.level_counts[: max_level + 1] += 1
				self.current_max[idx] = 0

	def metrics(self, prefix: str):
		metrics = {}
		lives = max(1, self.completed_lives)
		for lvl in range(self.n_levels):
			count = int(self.level_counts[lvl])
			pct = float(count / lives) if self.completed_lives > 0 else 0.0
			metrics[f"{prefix}/level_reached_count/level_{lvl}"] = count
			metrics[f"{prefix}/level_reached_pct/level_{lvl}"] = pct
		metrics[f"{prefix}/lives"] = int(self.completed_lives)
		metrics[f"{prefix}/screen_complete_count"] = int(self.screen_completions)
		screen_pct = float(self.screen_completions / lives) if self.completed_lives > 0 else 0.0
		metrics[f"{prefix}/screen_complete_pct"] = screen_pct
		return metrics


class LevelCompletionLogger(BaseCallback):
	def __init__(self, n_levels: int = 7, prefix: str = "train"):
		super().__init__()
		self.prefix = prefix
		self.stats = LifeStatsAggregator(n_levels=n_levels)

	def _on_training_start(self) -> None:
		self.stats.reset_state(self.training_env.num_envs)

	def _on_step(self) -> bool:
		self.stats.process_step(self.locals.get("infos"), self.locals.get("dones"))
		self._record()
		return True

	def _record(self):
		for key, val in self.stats.metrics(self.prefix).items():
			self.logger.record(key, val)


class GradientInspectionCallback(BaseCallback):
	def __init__(self, verbose=0):
		super(GradientInspectionCallback, self).__init__(verbose)

	def _on_step(self) -> bool:
		policy_net = self.model.policy

		grad_norms = []
		for name, param in policy_net.named_parameters():
			if param.grad is not None:
				grad_norm = param.grad.norm().item()
				grad_norms.append(grad_norm)
				self.logger.record(f"gradients/{name}_norm", grad_norm)
		if grad_norms:
			avg_grad_norm = sum(grad_norms) / len(grad_norms)
			self.logger.record("gradients/avg_norm", avg_grad_norm)

		return True


def build_callbacks(config: dict, checkpoints_dir: Path) -> CallbackList:
	eval_env = make_eval_vec_env(config)
	save_freq = int(config["model"]["save_freq"])
	n_envs = int(config["env"]["n_envs"])
	target_eval_freq = config["eval"]["eval_freq"]
	if target_eval_freq is None:
		target_eval_freq = save_freq
	target_eval_freq = int(target_eval_freq)
	adjusted_eval_freq = max(target_eval_freq // n_envs, 1)
	n_eval_episodes = int(config["eval"]["n_eval_episodes"])
	video_every = int(config["eval"]["video_every"])
	video_ctrl_cb = EvalVideoControlCallback(eval_env, video_every)

	best_dir = Path(checkpoints_dir) / "best"
	log_path = Path(config["monitor_dir"]) / "eval"
	
	eval_cb = CustomEvalWithStats(
		eval_env=eval_env,
		best_model_save_path=str(best_dir),
		log_path=str(log_path),
		eval_freq=adjusted_eval_freq,
		n_eval_episodes=n_eval_episodes,
		deterministic=True,
		verbose=1,
		callback_after_eval=video_ctrl_cb,
		n_levels=config["env"]["wrappers"].get("max_levels", 7),
	)
	wandb_cb = WandbCallback(
		config=config,
		model_save_path=str(checkpoints_dir),
		model_save_freq=save_freq,
		gradient_save_freq=int(config["wandb"]["gradient_save_freq"]),
	)
	gradient_inspection_cb = GradientInspectionCallback(verbose=1)
	level_logger_cb = LevelCompletionLogger(n_levels=config["env"]["wrappers"].get("max_levels", 7))
	return CallbackList([eval_cb, wandb_cb, gradient_inspection_cb, level_logger_cb])


class CustomEvalWithStats(EvalCallback):
	def __init__(self, *args, run_before_training: bool = True, **kwargs):
		n_levels = kwargs.pop("n_levels", 7)
		callback_after_eval = kwargs.pop("callback_after_eval", None)
		super().__init__(*args, callback_after_eval=callback_after_eval, **kwargs)
		self.run_before_training = run_before_training
		self.run_before_training_done = False
		self.stats = LifeStatsAggregator(n_levels=n_levels)
		self.evaluations_timesteps = []
		self.evaluations_results = []
		self.evaluations_lengths = []
		# Ensure attr exists even if base class signature changes
		self.callback_after_eval = callback_after_eval

	def _on_training_start(self) -> None:
		super()._on_training_start()
		if self.run_before_training and not self.run_before_training_done:
			prev_n_calls = self.n_calls
			self.n_calls = self.eval_freq if self.eval_freq > 0 else 1
			self._run_eval_and_log()
			self.n_calls = prev_n_calls
			self.run_before_training_done = True

	def _on_step(self) -> bool:
		if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
			self._run_eval_and_log()
		return True

	def _run_eval_and_log(self):
		self.stats.reset_state(1)
		episode_rewards = []
		episode_lengths = []

		while len(episode_rewards) < self.n_eval_episodes:
			obs = self.eval_env.reset()
			done = False
			ep_rew = 0.0
			ep_len = 0
			while not done:
				action, _ = self.model.predict(obs, deterministic=self.deterministic)
				obs, rewards, dones, infos = self.eval_env.step(action)
				self.stats.process_step(infos, dones)
				ep_rew += float(rewards[0])
				ep_len += 1
				done = bool(dones[0])
			episode_rewards.append(ep_rew)
			episode_lengths.append(ep_len)

		mean_reward = float(np.mean(episode_rewards))
		std_reward = float(np.std(episode_rewards))
		mean_ep_length = float(np.mean(episode_lengths))

		self.last_mean_reward = mean_reward
		self.last_mean_ep_length = mean_ep_length

		if self.best_mean_reward is None or mean_reward > self.best_mean_reward:
			if self.verbose > 0:
				logger.info(f"New best mean reward: {mean_reward:.2f}")
			self.best_mean_reward = mean_reward
			if self.best_model_save_path is not None:
				save_dir = Path(self.best_model_save_path)
				save_dir.mkdir(parents=True, exist_ok=True)
				save_path = save_dir / "best_model"
				self.model.save(save_path)
			if self.callback_on_new_best is not None:
				self.callback_on_new_best.on_step()

		self.logger.record("eval/mean_reward", mean_reward)
		self.logger.record("eval/mean_reward_std", std_reward)
		self.logger.record("eval/mean_ep_length", mean_ep_length)

		self.evaluations_timesteps.append(self.model.num_timesteps)
		self.evaluations_results.append(episode_rewards)
		self.evaluations_lengths.append(episode_lengths)

		if self.log_path is not None:
			log_dir = Path(self.log_path)
			log_dir.mkdir(parents=True, exist_ok=True)
			np.savez(
				log_dir / "evaluations",
				timesteps=self.evaluations_timesteps,
				results=self.evaluations_results,
				ep_lengths=self.evaluations_lengths,
			)

		for key, val in self.stats.metrics("eval").items():
			self.logger.record(key, val)

		if self.callback_after_eval is not None:
			self.callback_after_eval.on_step()


if __name__ == "__main__":
	config = load_config()
	print("Config: ", config)
