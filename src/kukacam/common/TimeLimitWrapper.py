import gym
from pybullet_envs.bullet.racecarZEDGymEnv import RacecarZEDGymEnv


class TimeLimitWrapper(gym.Wrapper):
  """
  :param env: (gym.env) gym environment that will be wrapped
  :param max_steps: (int) max number of steps per episode
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


if __name__ == "__main__":
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

  env.close()
