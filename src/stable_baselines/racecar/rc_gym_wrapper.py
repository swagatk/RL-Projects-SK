import gym
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pybullet_envs.bullet.racecarZEDGymEnv import RacecarZEDGymEnv

class RaceCarCustomWrapper4pt(gym.Wrapper):
    """
    Custom Gym Wrapper for PyTorch (Stable Baselines)
    :param env: (gym.Env)   Gym environment that will be wrapped
    - Downsample the image observation to a square image
    - Uses only RGB channels of the image
    - Limits the maximum number of steps to 20
    - Converts Channel last (HxWxC) to Channel first (CxHxW) format  
    - normalize the pixels to (0,1)
    """

    def __init__(self, env, shape, max_steps=100):

        assert isinstance(env.observation_space, gym.spaces.Box), \
            "Valid for continuous observation spaces of type gym.spaces.Box"

        super(RaceCarCustomWrapper4pt, self).__init__(env)
        self.max_steps = max_steps
        self.current_step = 0

        # create square image with channel first
        if isinstance(shape, int):
            shape = (shape, shape)
        assert all(x > 0 for x in shape), shape
        self.shape = tuple(shape)

        obs_shape = (3,) + self.shape     # square shape: (C, H, H)

        self.observation_space = gym.spaces.Box(low=0, high=255, shape=obs_shape, dtype=np.uint8)

    def _modify_obsvn(self, obs):
        rgb_obs = cv2.cvtColor(obs, cv2.COLOR_RGBA2RGB)     # convert from RGBA to RGB
        new_obs = cv2.resize(rgb_obs, self.shape, interpolation=cv2.INTER_AREA)     # make it square
        new_obs = np.transpose(new_obs, (2, 0, 1)) # channel first format
        new_obs = np.asarray(new_obs, dtype=np.float32) / 255.0 # normalize the pixels
        return new_obs

    def reset(self):
        """
        Convert RGBA image to RGB image
        Resize the image
        Convert into Channel-First format
        Normalize the pixels
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
        info['observation_shape'] = np.shape(new_obs)
        return new_obs, reward, done, info

#######################################
if __name__ == "__main__":

    TIME_LIMIT = True    
    OBSV_RESIZE = False

    # Test the wrapper
    env = RacecarZEDGymEnv(isDiscrete=False, renders=False)
    env = RaceCarCustomWrapper4pt(env, shape=40, max_steps=20)

    print('observation shape:', env.observation_space.shape)

    for ep in range(10):
        obs = env.reset()
        print(np.shape(obs))
        obs_img = np.moveaxis(obs, 0, 2)
        plt.imshow(obs_img)
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

