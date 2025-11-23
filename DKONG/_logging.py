from __future__ import annotations
import wandb
import yaml
import logging
from pathlib import Path
from wandb.integration.sb3 import WandbCallback as SB3WandbCallback
from stable_baselines3.common.callbacks import EvalCallback, CallbackList, BaseCallback

from environment.env import make_eval_vec_env


DKONG_ROOT = Path(__file__).resolve().parents[0]
CONFIG_PATH = DKONG_ROOT / "config.yml"
logger = logging.getLogger(__name__)

def load_config() -> dict:
	with open(CONFIG_PATH, "r") as f:
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


class CustomEvalCallback(EvalCallback):
	def __init__(self, *args, run_before_training: bool = True, **kwargs):
		super().__init__(*args, **kwargs)
		self.run_before_training = run_before_training
		self.run_before_training_done = False

	def _on_training_start(self) -> None:
		# run evaluation before training starts
		super()._on_training_start()
		if self.run_before_training and not self.run_before_training_done:
			# temporarily fake n_calls so that _on_step executes once
			prev_n_calls = self.n_calls
			self.n_calls = self.eval_freq if self.eval_freq > 0 else 1
			self._on_step()
			self.n_calls = prev_n_calls
			self.run_before_training_done = True


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
	
	eval_cb = CustomEvalCallback(
		eval_env=eval_env,
		best_model_save_path=str(best_dir),
		log_path=str(log_path),
		eval_freq=adjusted_eval_freq,
		n_eval_episodes=n_eval_episodes,
		deterministic=True,
		verbose=1,
		callback_after_eval=video_ctrl_cb,
	)
	wandb_cb = WandbCallback(
		config=config,
		model_save_path=str(checkpoints_dir),
		model_save_freq=save_freq,
		gradient_save_freq=int(config["wandb"]["gradient_save_freq"]),
	)
	return CallbackList([eval_cb, wandb_cb])


if __name__ == "__main__":
	config = load_config()
	print("Config: ", config)
