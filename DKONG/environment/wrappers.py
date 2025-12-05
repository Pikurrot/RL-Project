from __future__ import annotations
import gymnasium as gym
import numpy as np
import cv2
from gymnasium.wrappers import ResizeObservation as GymResizeObservation

from schedulers import build_step_schedule


# Reduce action space to only minimal actions
class MinimalActionSpace(gym.ActionWrapper):
	def __init__(self, env: gym.Env, config: dict):
		super().__init__(env)
		wcfg = config["env"]["wrappers"]["minimal_action_space"]
		if isinstance(wcfg, dict) and "minimal_actions" in wcfg:
			self.minimal_actions = wcfg["minimal_actions"]
		else:
			self.minimal_actions = config["env"]["wrappers"]["minimal_action_space"]
		self.action_space = gym.spaces.Discrete(len(self.minimal_actions))

	def action(self, action: int) -> int:
		return self.minimal_actions[action]


# Automatically press FIRE whenever a new life begins so gameplay resumes
class FireAtStart(gym.Wrapper):
	def __init__(self, env: gym.Env):
		super().__init__(env)
		action_meanings = self.env.unwrapped.get_action_meanings()
		if "FIRE" not in action_meanings:
			raise ValueError("FireAtStart requires an environment with a FIRE action")
		self._fire_action = action_meanings.index("FIRE")
		self._lives: int | None = None
		self._pending_fire_frames: int = 0
		self._fire_repeat_steps = 12  # empirically enough frames to skip the death animation

	def reset(self, **kwargs):
		obs, info = self.env.reset(**kwargs)
		self._lives = self._extract_lives(info)
		self._pending_fire_frames = 0
		obs, info = self._fire_until_running(obs, info, **kwargs)
		return obs, info

	def step(self, action: int):
		self._fire_next_step_if_needed()
		obs, reward, terminated, truncated, info = self.env.step(action)
		lives = self._extract_lives(info)

		life_lost = (
			self._lives is not None
			and lives is not None
			and lives > 0
			and lives < self._lives
		)

		game_over = (
			self._lives is not None
			and lives is not None
			and lives == 0
			and not terminated
			and not truncated
		)

		if life_lost and not terminated and not truncated:
			self._pending_fire_frames = self._fire_repeat_steps

		if game_over:
			terminated = True
			self._pending_fire_frames = 0
			info = dict(info)
			info["fire_wrapper_forced_game_over"] = True

		self._lives = lives
		return obs, reward, terminated, truncated, info

	def _fire_until_running(self, obs, info, **kwargs):
		# Some Atari envs require the FIRE action to exit the ready screen.
		obs, _, terminated, truncated, info = self.env.step(self._fire_action)
		while terminated or truncated:
			obs, info = self.env.reset(**kwargs)
			obs, _, terminated, truncated, info = self.env.step(self._fire_action)
		self._lives = self._extract_lives(info)
		self._pending_fire_frames = 0
		return obs, info

	def _fire_next_step_if_needed(self):
		while self._pending_fire_frames > 0:
			obs, _, terminated, truncated, info = self.env.step(self._fire_action)
			self._lives = self._extract_lives(info)
			self._pending_fire_frames -= 1

			if terminated or truncated:
				self._pending_fire_frames = 0
				return

	def _extract_lives(self, info: dict) -> int | None:
		if "lives" in info:
			return info["lives"]
		ale = getattr(self.env.unwrapped, "ale", None)
		if ale is not None:
			return ale.lives()
		return None


# Add channel dimension to observation (for the CNN model)
class AddChannelDim(gym.ObservationWrapper):
	def __init__(self, env: gym.Env):
		super().__init__(env)
		s = self.observation_space.shape
		self.observation_space = gym.spaces.Box(low=0, high=255, shape=(1, *s), dtype=np.uint8)

	def observation(self, observation: np.ndarray) -> np.ndarray: # (H, W)
		return np.expand_dims(observation, axis=0).astype(np.uint8) # (1, H, W)


# Scale observation to [0, 1] (must be after AddChannelDim)
class ScaleObservation(gym.ObservationWrapper):
	def __init__(self, env: gym.Env):
		super().__init__(env)
		self.observation_space = gym.spaces.Box(low=0, high=1, shape=self.observation_space.shape, dtype=np.float32)

	def observation(self, observation: np.ndarray) -> np.ndarray:
		return observation.astype(np.float32) / 255.0


# Grayscale observation (must be before AddChannelDim)
class GrayscaleObservation(gym.ObservationWrapper):
	def __init__(self, env: gym.Env):
		super().__init__(env)
		shape = self.observation_space.shape
		out_shape = (shape[0], shape[1])
		self.observation_space = gym.spaces.Box(low=0, high=255, shape=out_shape, dtype=np.uint8)

	def observation(self, observation: np.ndarray) -> np.ndarray: # (H, W, 3)
		return cv2.cvtColor(observation, cv2.COLOR_RGB2GRAY) # (H, W)


# Resize observation (must be before AddChannelDim)
class ResizeObservation(GymResizeObservation):
	def __init__(self, env: gym.Env, config: dict):
		wcfg = config["env"]["wrappers"]["resize_observation"]
		self.resize_observation = tuple(wcfg["size"]) if isinstance(wcfg, dict) else tuple(wcfg)
		super().__init__(env, self.resize_observation)


# Death penalty
class DeathPenalty(gym.Wrapper):
	def __init__(self, env: gym.Env, config: dict):
		super().__init__(env)
		my_config = config["env"]["wrappers"]["death_penalty"]
		self.death_penalty = my_config["value"] if isinstance(my_config, dict) else my_config

	def step(self, action: int):
		obs, reward, terminated, truncated, info = self.env.step(action)
		if info.get("fire_wrapper_forced_game_over", False):
			reward += self.death_penalty

		if terminated or truncated:
			reward += self.death_penalty
		
		return obs, reward, terminated, truncated, info


def mario_position(obs):
	redness = obs[:, :, 0].astype(np.int16) - np.maximum(obs[:, :, 1], obs[:, :, 2]).astype(np.int16)
	max_redness = np.max(redness)
	most_reddish = (redness == max_redness)
	kernel = np.ones((2, 2), np.uint8)
	most_reddish = cv2.morphologyEx(most_reddish.astype(np.uint8), cv2.MORPH_OPEN, kernel)
	center_of_mass = np.argwhere(most_reddish)
	if center_of_mass.size == 0:
		return np.array([np.nan, np.nan])
	center_of_mass = np.mean(center_of_mass, axis=0)
	return center_of_mass

def distance_to_ladders(mario_pos):
	def get_level(center_of_mass):
		levels_height = [25, 25, 35, 25, 25, 25, 25]
		levels_y_lst = [170, 145, 110, 85, 60, 35, 10]
		for i, y in enumerate(levels_y_lst):
			if center_of_mass[0] > y:
				return i
		# if reaches here, probably mario is not in the obs (when died)
		return 0
	
	def get_ladders_from_level(level):
		ladders_x_coords = [[110], [50, 83], [110], [50, 71], [110], [34, 79], []]
		return ladders_x_coords[level]

	def get_distance_to_ladders(center_of_mass, ladders_x_coords):
		if len(ladders_x_coords) == 0:
			return [np.nan]
		distance = []
		for ladder in ladders_x_coords:
			distance.append(abs(center_of_mass[1] - ladder))
		return distance
	
	level = get_level(mario_pos)
	ladders_x_coords = get_ladders_from_level(level)
	distance = get_distance_to_ladders(mario_pos, ladders_x_coords)
	return distance


# Reward for climbing a ladder (must be before ResizeObservation)
class LadderClimbReward(gym.Wrapper):
	def __init__(self, env: gym.Env, config: dict):
		super().__init__(env)
		my_config = config["env"]["wrappers"]["ladder_climb_reward"]
		self.reward_per_pixel = my_config["reward_per_pixel"]
		self.require_up_action = my_config["require_up_action"]
		self.max_bonus = my_config["max_bonus"]
		self.prev_obs = None

	def reset(self, **kwargs):
		obs, info = self.env.reset(**kwargs)
		self.prev_obs = obs
		return obs, info

	def step(self, action: int):
		if self.prev_obs is None:
			self.prev_obs, _ = self.env.reset()

		y_prev = mario_position(self.prev_obs)[0]
		obs, reward, terminated, truncated, info = self.env.step(action)
		y = mario_position(obs)[0]

		valid_action = (not self.require_up_action) or (action == 2)
		if not np.isnan(y_prev) and not np.isnan(y) and valid_action:
			pixel_gain = max(0.0, y_prev - y)
			bonus = self.reward_per_pixel * pixel_gain
			if self.max_bonus is not None:
				bonus = np.clip(bonus, None, self.max_bonus)
			reward += bonus

		self.prev_obs = obs
		return obs, reward, terminated, truncated, info


# Penalty for distance to ladders (kept for compatibility; supports scheduling)
class DistanceToLaddersPenalty(gym.Wrapper):
	def __init__(self, env: gym.Env, config: dict):
		super().__init__(env)
		my_config = config["env"]["wrappers"]["ladder_distance_penalty"]
		self.per_pixel_penalty = my_config["per_pixel_penalty"]
		schedule_cfg = my_config["schedule"] if isinstance(my_config, dict) and "schedule" in my_config else None
		self.schedule = build_step_schedule(schedule_cfg)
		self.step_count = 0

	def reset(self, **kwargs):
		self.step_count = 0
		return self.env.reset(**kwargs)

	def step(self, action: int):
		obs, reward, terminated, truncated, info = self.env.step(action)
		mario_pos = mario_position(obs)
		distance = distance_to_ladders(mario_pos) or [0]
		if np.isnan(np.min(distance)):
			distance = 0
		else:
			distance = np.min(distance)
		scale = self.per_pixel_penalty
		if self.schedule is not None:
			scale = self.schedule.value(self.step_count)
		reward += scale * distance
		self.step_count += 1
		return obs, reward, terminated, truncated, info


class LadderDistancePotential(gym.Wrapper):
	"""
	Potential-based shaping: reward += scale * (gamma * Phi(s') - Phi(s)),
	where Phi(s) = -min ladder distance on current level.
	"""
	def __init__(self, env: gym.Env, config: dict):
		super().__init__(env)
		my_config = config["env"]["wrappers"]["ladder_distance_potential"]
		self.gamma = config["model"]["PPO"]["gamma"]
		self.base_scale = my_config["scale"]
		schedule_cfg = my_config["schedule"] if isinstance(my_config, dict) and "schedule" in my_config else None
		self.schedule = build_step_schedule(schedule_cfg)
		self.prev_potential = None
		self.step_count = 0

	def reset(self, **kwargs):
		obs, info = self.env.reset(**kwargs)
		self.prev_potential = self._potential(obs)
		self.step_count = 0
		return obs, info

	def step(self, action: int):
		obs, reward, terminated, truncated, info = self.env.step(action)
		potential = self._potential(obs)
		scale = self.base_scale
		if self.schedule is not None:
			scale = self.schedule.value(self.step_count)
		if self.prev_potential is not None and not np.isnan(self.prev_potential) and not np.isnan(potential):
			shaping = scale * (self.gamma * potential - self.prev_potential)
			reward += shaping
		self.prev_potential = potential
		self.step_count += 1
		return obs, reward, terminated, truncated, info

	def _potential(self, obs) -> float:
		mario_pos = mario_position(obs)
		distances = distance_to_ladders(mario_pos)
		min_distance = np.nanmin(distances)
		if np.isnan(min_distance):
			return 0.0
		return -min_distance


class LadderAlignmentBonus(gym.Wrapper):
	"""
	One-time bonuses when Mario aligns with a ladder and attempts to climb.
	"""
	def __init__(self, env: gym.Env, config: dict):
		super().__init__(env)
		my_config = config["env"]["wrappers"]["ladder_alignment_bonus"]
		self.bonus = my_config["bonus"]
		self.x_tolerance = my_config["x_tolerance"]
		self.cooldown_steps = my_config["cooldown_steps"]
		self.cooldown = 0

	def reset(self, **kwargs):
		self.cooldown = 0
		return self.env.reset(**kwargs)

	def step(self, action: int):
		obs, reward, terminated, truncated, info = self.env.step(action)
		if self.cooldown > 0:
			self.cooldown -= 1
			return obs, reward, terminated, truncated, info

		mario_pos = mario_position(obs)
		distances = distance_to_ladders(mario_pos)
		if np.isnan(np.min(distances)):
			return obs, reward, terminated, truncated, info

		if np.min(distances) <= self.x_tolerance and action == 2:  # aligned and pressing UP
			reward += self.bonus
			self.cooldown = self.cooldown_steps

		return obs, reward, terminated, truncated, info
