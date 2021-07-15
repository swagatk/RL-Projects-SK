"""
Gym Wrappers
"""
import gym
import numpy as np
from pybullet_envs.bullet.racecarZEDGymEnv import RacecarZEDGymEnv
from pybullet_envs.bullet.kuka_diverse_object_gym_env import KukaDiverseObjectEnv
from stable_baselines3.common.env_checker import check_env


class NormalizeObsvnWrapper(gym.Wrapper):
    """
    :param env: (gym.Env)   Gym environment that will be wrapped

    - normalizes the pixels between 0 to 1
    - Converts image format from HxWxC (channel-last) to CxHxW (channel-first)
    """

    def __init__(self, env):
        assert isinstance(env.observation_space, gym.spaces.Box), \
            "Valid for continuous observation spaces of type gym.spaces.Box"

        self._height = env.observation_space.shape[0]
        self._width = env.observation_space.shape[1]
        self._channels = env.observation_space.shape[2]

        env.observation_space = gym.spaces.Box(low=0, high=255,
                                               shape=(self._channels,
                                                      self._height,
                                                      self._width))
        # call the parent constructor so that we can access self.env
        super(NormalizeObsvnWrapper, self).__init__(env)

    def _modify_obsvn(self, obs):
        new_obs = np.transpose(obs, (2, 0, 1))
        new_obs = np.asarray(new_obs, dtype=np.float32) / 255.0
        return new_obs

    def reset(self):
        """
        Convert Images from HxWxC format to CxHxW
        Normalize the pixels between 0 and 1.0
        """
        return self._modify_obsvn(self.env.reset())

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        new_obs = self._modify_obsvn(obs)
        info['channel_first'] = True
        info['nomalize pixel'] = True
        return new_obs, reward, done, info


class TimeLimitWrapper(gym.Wrapper):
  """
  :param env: (gym.env) gym environment that will be wrapped
  :param max_steps: (int) max number of steps per episode
  Limits the maximum number of steps per episode
  """
  def __init__(self, env, max_steps=100):
    # call the parent constructor, so we can access self.env later
    super(TimeLimitWrapper, self).__init__(env)
    self.max_steps = max_steps
    # counter of steps per episode
    self.current_step = 0

  def reset(self):
    """
    Reset the environment
    """
    self.current_step = 0
    return self.env.reset()

  def step(self, action):
    """
    :param action: ([float] or int) Action taken by the agent
    :return: (np.ndarray, float, bool, dict) observation, reward, done and info
    """
    self.current_step += 1
    obs, reward, done, info = self.env.step(action)
    if self.current_step >= self.max_steps:
      done = True
      # update info dict to signal that the limit was exceeded
      info['time_limit_reached'] = True
    return obs, reward, done, info


if __name__ == '__main__':

    Normalize = True
    TimeLimit = False

    # Testing the above wrapper
    import pybullet as p

    if Normalize:
        env = NormalizeObsvnWrapper(KukaDiverseObjectEnv(maxSteps=20, isDiscrete=False, renders=False,
                                                         removeHeightHack=False))
        check_env(env)
        obs = env.reset()
        print('shape of observation space:', env.observation_space.shape)
        print('shape of observation:', np.shape(obs))

    elif TimeLimit:
        # Test the wrapper
        env = RacecarZEDGymEnv(isDiscrete=False, renders=False)
        env = TimeLimitWrapper(env, max_steps=20)

        for ep in range(5):
            obs = env.reset()
            t = 0
            ep_reward = 0
            while True:
                action = env.action_space.sample()
                next_obs, reward, done, info = env.step(action)
                t += 1
                ep_reward += reward

                if done:
                    print('Episode:{}, steps:{}, Score:{}'.format(ep, t, ep_reward))
                    break


