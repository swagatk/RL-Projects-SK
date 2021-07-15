import gym
import numpy as np
from pybullet_envs.bullet.racecarZEDGymEnv import RacecarZEDGymEnv
import matplotlib.pyplot as plt
import cv2


class ObsvnResizeTimeLimitWrapper(gym.Wrapper):
    """
    :param env: (gym.Env)   Gym environment that will be wrapped
    - Downsample the image observation to a square image
    - Uses only RGB channels of the image
    - Limits the maximum number of steps to 20
    """

    def __init__(self, env, shape, max_steps=100):
        super(ObsvnResizeTimeLimitWrapper, self).__init__(env)
        self.max_steps = max_steps
        self.current_step = 0

        if isinstance(shape, int):
            shape = (shape, shape)
        assert all(x > 0 for x in shape), shape
        self.shape = tuple(shape)

        obs_shape = self.shape + (3,)

        self.observation_space = gym.spaces.Box(low=0, high=255, shape=obs_shape, dtype=np.uint8)

    def _modify_obsvn(self, obs):
        rgb_obs = cv2.cvtColor(obs, cv2.COLOR_RGBA2RGB)
        new_obs = cv2.resize(rgb_obs, self.shape, interpolation=cv2.INTER_AREA)
        #new_obs = np.asarray(new_obs, dtype=np.float32) / 255.0
        return new_obs

    def reset(self):
        """
        Convert RGBA image to RGB image
        Resize the image
        """
        self.current_step = 0
        return self._modify_obsvn(self.env.reset())

    def step(self, action):
        self.current_step += 1
        obs, reward, done, info = self.env.step(action)
        if self.current_step > self.max_steps:
            done = True
        new_obs = self._modify_obsvn(obs)
        info['channel_first'] = True
        info['nomalize_pixel'] = True
        info['time_limit_reached'] = True
        return new_obs, reward, done, info


if __name__ == "__main__":
    # Test the wrapper
    env = RacecarZEDGymEnv(isDiscrete=False, renders=False)
    env = ObsvnResizeTimeLimitWrapper(env, shape=40, max_steps=20)

    print('observation shape:', env.observation_space.shape)

    for ep in range(10):
        obs = env.reset()
        plt.imshow(obs)
        plt.show()
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

