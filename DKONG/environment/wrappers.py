from __future__ import annotations
import gymnasium as gym
import numpy as np


# Reduce action space to only minimal actions
class MinimalActionSpace(gym.ActionWrapper):
	def __init__(self, env: gym.Env, config: dict):
		super().__init__(env)
		self.minimal_actions = config["env"]["minimal_actions"]
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


# Scale observation to [0, 1]
class ScaleObservation(gym.ObservationWrapper):
	def __init__(self, env: gym.Env):
		super().__init__(env)
		self.observation_space = gym.spaces.Box(low=0, high=1, shape=self.observation_space.shape, dtype=np.float32)

	def observation(self, observation: np.ndarray) -> np.ndarray:
		return observation.astype(np.float32) / 255.0
