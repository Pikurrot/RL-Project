from __future__ import annotations
import wandb
import yaml
import logging
from pathlib import Path
from wandb.integration.sb3 import WandbCallback as SB3WandbCallback


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


if __name__ == "__main__":
	config = load_config()
	print("Config: ", config)
