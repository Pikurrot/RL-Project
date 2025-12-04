from __future__ import annotations
import math
from typing import Callable, Optional, Union, Dict, Any

Number = Union[float, int]


def _clamp01(x: float) -> float:
	x = max(0.0, min(1.0, x))
	return x


def _progress(step: int, total_steps: int) -> float:
	if total_steps <= 0:
		return 1.0
	return _clamp01(step / float(total_steps))


def linear_interp(start: float, end: float, t: float) -> float:
	return start + (end - start) * t


def cosine_interp(start: float, end: float, t: float) -> float:
	# Smooth start/end with half-cosine
	return end + (start - end) * 0.5 * (1.0 + math.cos(math.pi * t))


class StepSchedule:
	"""
	Simple step-based scheduler for wrappers that don't receive SB3 progress callbacks.
	"""
	def __init__(self, start: Number, end: Number, total_steps: int, kind: str = "linear"):
		self.start = float(start)
		self.end = float(end)
		self.total_steps = int(total_steps)
		self.kind = kind

	def value(self, step: int) -> float:
		t = _progress(step, self.total_steps)
		if self.kind == "linear":
			return linear_interp(self.start, self.end, t)
		if self.kind == "cosine":
			return cosine_interp(self.start, self.end, t)
		# constant fallback
		return self.end


def build_step_schedule(spec: Optional[Dict[str, Any]]) -> Optional[StepSchedule]:
	if spec is None:
		return None
	if isinstance(spec, (int, float)):
		return StepSchedule(spec, spec, total_steps=1, kind="constant")

	kind = spec.get("type", "constant")
	if kind == "constant":
		val = spec.get("value", spec.get("start", 0.0))
		return StepSchedule(val, val, total_steps=1, kind="constant")

	start = spec.get("start", 0.0)
	end = spec.get("end", start)
	total_steps = int(spec.get("total_steps", 1))
	return StepSchedule(start, end, total_steps, kind=kind)


def build_sb3_schedule(spec: Optional[Union[Number, Dict[str, Any]]]) -> Union[Number, Callable[[float], float]]:
	"""
	Convert a config spec into an SB3-compatible schedule.
	SB3 expects a callable(progress_remaining) or a float.
	"""
	if spec is None:
		return None
	if isinstance(spec, (int, float)):
		return float(spec)

	kind = spec.get("type", "constant")
	if kind == "constant":
		return float(spec.get("value", spec.get("start", 0.0)))

	start = float(spec.get("start", 0.0))
	end = float(spec.get("end", start))

	if kind == "linear":
		def schedule(progress_remaining: float) -> float:
			return linear_interp(start, end, 1.0 - progress_remaining)
	elif kind == "cosine":
		def schedule(progress_remaining: float) -> float:
			return cosine_interp(start, end, 1.0 - progress_remaining)
	else:
		# fallback to constant
		def schedule(progress_remaining: float) -> float:
			return end

	return schedule

