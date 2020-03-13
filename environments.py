"""Gym environments."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gym
from gym.envs.classic_control import acrobot
from gym.envs.classic_control import cartpole
from gym.envs.classic_control import continuous_mountain_car
from gym.envs.classic_control import mountain_car
from gym.envs.registration import register
import numpy as np


class PartialCartPoleEnv(cartpole.CartPoleEnv):
  """Cartpole environment with velocity components removed."""

  def __init__(self):
    super(PartialCartPoleEnv, self).__init__()
    # Increase the observation range.
    high = np.array([
        self.x_threshold * 2,
        self.theta_threshold_radians * 2,
    ])
    self.observation_space = gym.spaces.Box(-high, high)

  def _mask_observation(self, observation):
    """Returns a new observation without velocity components."""
    return observation[[0, 2]]  # Remove index 1 and 3.

  def reset(self):
    """Resets the internal state."""
    observation = super(PartialCartPoleEnv, self).reset()
    return self._mask_observation(observation)

  def step(self, action):
    """Returns a new state after an action is taken."""
    step_out = super(PartialCartPoleEnv, self).step(action)
    observation, reward, done, info = step_out
    return self._mask_observation(observation), reward, done, info


register(
    id='PartialCartPole-v0',
    entry_point=PartialCartPoleEnv,
    max_episode_steps=200,
    reward_threshold=195.0,
)

register(
    id='PartialCartPole-v1',
    entry_point=PartialCartPoleEnv,
    max_episode_steps=500,
    reward_threshold=475.0,
)


class PartialMountainCarEnv(mountain_car.MountainCarEnv):
  """Mountain car environment without velocity."""

  def __init__(self, goal_velocity=0):
    super(PartialMountainCarEnv, self).__init__(goal_velocity)
    self.low = np.array([self.min_position])
    self.high = np.array([self.max_position])
    self.observation_space = gym.spaces.Box(
        self.low, self.high, dtype=np.float32)

  def _mask_observation(self, observation):
    """Returns a new observation without velocity."""
    return observation[[0]]  # Remove index 1

  def reset(self):
    """Resets the internal state."""
    observation = super(PartialMountainCarEnv, self).reset()
    return self._mask_observation(observation)

  def step(self, action):
    """Returns a new state after an action is taken."""
    step_out = super(PartialMountainCarEnv, self).step(action)
    observation, reward, done, info = step_out
    return self._mask_observation(observation), reward, done, info


register(
    id='PartialMountainCar-v0',
    entry_point=PartialMountainCarEnv,
    max_episode_steps=200,
    reward_threshold=-110.0,
)


class PartialContinuousMountainCarEnv(
    continuous_mountain_car.Continuous_MountainCarEnv):
  """Countinuous mountain car environment without velocity."""

  def __init__(self, goal_velocity=0):
    super(PartialContinuousMountainCarEnv, self).__init__(goal_velocity)
    self.low_state = np.array([self.min_position])
    self.high_state = np.array([self.max_position])
    self.observation_space = gym.spaces.Box(
        self.low_state, self.high_state, dtype=np.float32)

  def _mask_observation(self, observation):
    """Returns a new observation without velocity."""
    return observation[[0]]  # Remove index 1

  def reset(self):
    """Resets the internal state."""
    observation = super(PartialContinuousMountainCarEnv, self).reset()
    return self._mask_observation(observation)

  def step(self, action):
    """Returns a new state after an action is taken."""
    step_out = super(PartialContinuousMountainCarEnv, self).step(action)
    observation, reward, done, info = step_out
    return self._mask_observation(observation), reward, done, info


register(
    id='PartialContinuousMountainCar-v0',
    entry_point=PartialContinuousMountainCarEnv,
    max_episode_steps=999,
    reward_threshold=90.0,
)
