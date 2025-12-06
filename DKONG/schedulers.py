from __future__ import annotations
import math
from typing import Callable, Optional, Union, Dict, Any


def clamp01(x: float) -> float:
	x = max(0.0, min(1.0, x))
	return x

def progress(step: int, total_steps: int) -> float:
	# Fraction of the total steps completed
	if total_steps <= 0:
		return 1.0 # avoid division by zero
	return clamp01(step / float(total_steps))


def linear_interp(start: float, end: float, t: float) -> float:
	# Converts t [0, 1] to a value between start and end (linear)
	return start + (end - start) * t


def cosine_interp(start: float, end: float, t: float) -> float:
	# Converts t [0, 1] to a value between start and end (cosine)
	return end + (start - end) * 0.5 * (1.0 + math.cos(math.pi * t))


# Scheduler that adjusts a value over time based on the number of steps taken
class StepSchedule:
	def __init__(self, start: float, end: float, total_steps: int, kind: str = "linear"):
		self.start = float(start)
		self.end = float(end)
		self.total_steps = int(total_steps)
		self.kind = kind

	def value(self, step: int) -> float:
		t = progress(step, self.total_steps)
		if self.kind == "linear":
			return linear_interp(self.start, self.end, t)
		if self.kind == "cosine":
			return cosine_interp(self.start, self.end, t)
		return self.end


# Build a custom scheduler
def build_step_schedule(scheduler_config: Any) -> StepSchedule:
	if scheduler_config is None:
		return None

	kind = scheduler_config["type"]
	start = scheduler_config["start"]
	end = scheduler_config["end"]
	total_steps = scheduler_config["total_steps"]
	return StepSchedule(start, end, total_steps, kind=kind)


# Build a scheduler compatible with SB3
# Source: ChatGPT
def build_sb3_schedule(scheduler_config: Any) -> Union[float, Callable]:
	if scheduler_config is None:
		return None
	if isinstance(scheduler_config, (int, float)):
		return float(scheduler_config)

	kind = scheduler_config["type"]
	start = float(scheduler_config["start"])
	end = float(scheduler_config["end"])

	if kind == "linear":
		def schedule(progress_remaining: float) -> float:
			return linear_interp(start, end, 1.0 - progress_remaining)
	elif kind == "cosine":
		def schedule(progress_remaining: float) -> float:
			return cosine_interp(start, end, 1.0 - progress_remaining)
	else:
		def schedule(progress_remaining: float) -> float:
			return end

	return schedule
