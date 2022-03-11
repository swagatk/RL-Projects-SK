"""
Custom Gym Wrapper

11/03/2022: Added Frame Stack wrapper
"""
from collections import deque
import gym
import numpy as np
from pybullet_envs.bullet.racecarZEDGymEnv import RacecarZEDGymEnv
from pybullet_envs.bullet.kukaCamGymEnv import KukaCamGymEnv
import matplotlib.pyplot as plt
import cv2

#####################################33
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


#################################################3

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

#####################################3
## frame stacking wrapper
class FrameStackWrapper(gym.Wrapper):
    def __init__(self, env, k) -> None:
        super().__init__(env)

        self._k = k # number of frames to stack
        self._frames = deque([], maxlen=k)
        new_shape = env.observation_space.shape 
        self.observation_space = gym.spaces.Box(
            low=0, high=255, 
            shape=(new_shape[0], new_shape[1], 3 * k), 
            dtype=env.observation_space.dtype
            )

    def reset(self):
        obs = self.env.reset()[:,:,:3]  # (H, W, 3) 
        for _ in range(self._k):
            self._frames.append(obs)
        return self._get_obs()

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self._frames.append(obs[:,:,:3])    # store only the first 3 channels of the observation
        return self._get_obs(), reward, done, info

    def _get_obs(self):
        assert len(self._frames) == self._k
        return np.dstack(self._frames)  # (H, W, K)

    def display_stacked_obs(self, stacked_obs=None, file_name=None):
        if stacked_obs is None:
            stacked_obs = self._get_obs()
        else:
            assert len(stacked_obs.shape) == 3, "stacked_obs must be 3-dimensional"
            #assert stacked_obs.shape[2] == self._k, "stacked_obs must have shape (H, W, K)"
        

        assert(stacked_obs.shape[2] % 3 == 0), "stacked_img must have 3x channels"
        image_list = np.dsplit(stacked_obs, int(stacked_obs.shape[2]//3))      # split the stacked image into sections of 3 channels

        rows = int(np.ceil(len(image_list) / 2)) # upper bound for the number of rows
        cols = int(np.ceil(len(image_list) / 2)) # upper bound for the number of columns
        fig, axs = plt.subplots(rows, cols, figsize=(10, 10))
        fig.suptitle('Stacked Images', fontsize=12)
        for i in range(rows):
            for j in range(cols):
                k = i*cols+j
                if k < len(image_list):
                    axs[i, j].imshow(image_list[k])
                    axs[i, j].axis('off')
                    axs[i, j].set_title('Frame {}'.format(k))
        fig.tight_layout()
        plt.show()
        if file_name is not None:
            plt.savefig(file_name)
        

#######################################
if __name__ == "__main__":

    TIME_LIMIT = False    
    OBSV_RESIZE = False
    STACK_FRAMES = True

    # Test the wrapper
    # env = RacecarZEDGymEnv(isDiscrete=False, renders=False)
    env = KukaCamGymEnv(isDiscrete=False, renders=False)
    if OBSV_RESIZE: 
        env = ObsvnResizeTimeLimitWrapper(env, shape=40, max_steps=20)
    elif TIME_LIMIT:
        env = TimeLimitWrapper(env, max_steps=20)
    elif STACK_FRAMES:
        env = FrameStackWrapper(env, k=4)
    print('observation shape:', env.observation_space.shape)

    for ep in range(2):
        obs = env.reset()

        print('shape of obs:', np.shape(obs))
        if STACK_FRAMES:
            env.display_stacked_obs(obs, file_name='stacked_obs_{}.png'.format(ep))
        else:
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

