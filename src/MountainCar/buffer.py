import numpy as np

class ReplayBuffer():
    "Buffer to store environment transitions"
    def __init__(self,
                 obs_shape,
                 action_shape,
                 capacity) -> None:

        self.capacity = capacity
        self.obses = np.empty((capacity, *obs_shape), dtype=np.float32)
        self.next_obses = np.empty((capacity, *obs_shape), dtype=np.float32)
        self.actions = np.empty((capacity, *action_shape), dtype=np.float32)
        self.rewards = np.empty((capacity, 1), dtype=np.float32)
        self.dones = np.empty((capacity, 1), dtype=np.float32)

        self.idx = 0
        self.last_save = 0
        self.full = False


    def add(self, obs, action, reward, next_obs, done):
        np.copyto(self.obses[self.idx], obs)
        np.copyto(self.actions[self.idx], action)
        np.copyto(self.rewards[self.idx], reward)
        np.copyto(self.next_obses[self.idx], next_obs)
        np.copyto(self.dones[self.idx], done)

        self.idx = (self.idx + 1) % self.capacity
        self.full = self.full or self.idx == 0


    def sample(self, batch_size=24):
        idxs = np.random.randint(
            0, self.capacity if self.full else self.idx, size=batch_size
        )

        obses = self.obses[idxs]
        next_obses = self.next_obses[idxs]
        actions = self.actions[idxs]
        rewards = self.rewards[idxs]
        dones = self.dones[idxs]

        return obses, actions, rewards, next_obses, dones


    def __getitem__(self, index):
      if index >= 0 and index < self.capacity if self.full else self.idx:
        obs = self.obses[index]
        action = self.actions[index]
        reward = self.rewards[index]
        next_obs = self.next_obses[index]
        done = self.dones[index]
        return obs, action, reward, next_obs, done
      else:
        raise ValueError('Index is out of range')

    def __len__(self):
        return self.capacity if self.full else self.idx
