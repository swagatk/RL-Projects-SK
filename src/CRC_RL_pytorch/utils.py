import torch 
import numpy as np 
from torch.utils.data import Dataset 
import json
import random
import os

def preprocess_obs(obs, bits=5):
    """Preprocessing image, see https://arxiv.org/abs/1807.03039."""
    bins = 2 ** bits
    assert obs.dtype == torch.float32
    if bits < 8:
        obs = torch.floor(obs / 2 ** (8 - bits))
    obs = obs / bins
    obs = obs + torch.rand_like(obs) / bins
    obs = obs - 0.5
    return obs


def set_seed_everywhere(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def soft_update_params(net, target_net, tau):
    for param, target_param in zip(net.parameters(), target_net.parameters()):
        target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)


def load_config(key=None):
    #path = os.path.join('/home/swagat/GIT/CRC_RL', 'data.cfg')
    path = os.path.join('.', 'data.cfg')
    with open(path) as f:
        data = json.load(f)
    return data[key] if key is not None else data

class eval_mode(object):
    ''' It turns off the training flag for the objects'''
    def __init__(self, *models) -> None:
        self.models = models
        self.prev_states = []

    def __enter__(self):
        for model in self.models:
            self.prev_states.append(model.training)
            model.train(False)

    def __exit__(self, *args):
        for model, state in zip(self.models, self.prev_states):
            model.train(state)      # restore the original state
        return False  # not really required. 

class ReplayBuffer(Dataset):
    "Buffer to store environment transitions"
    def __init__(self,
                 obs_shape,
                 action_shape,
                 capacity,
                 batch_size,
                 device,
                 image_size=84,
                 transform1=None,
                 transform2=None,
                 ) -> None:
        super().__init__()

        self.capacity = capacity
        self.batch_size = batch_size
        self.device = device 
        self.image_size = image_size
        self.transform1 = transform1
        self.transform2 = transform2
        obs_dtype = np.uint8  # pixel obs are stored as uint8 

        self.obses = np.empty((capacity, *obs_shape), dtype=obs_dtype)
        self.next_obses = np.empty((capacity, *obs_shape), dtype=obs_dtype)
        self.actions = np.empty((capacity, *action_shape), dtype=np.float32)
        self.rewards = np.empty((capacity, 1), dtype=np.float32)
        self.not_dones = np.empty((capacity, 1), dtype=np.float32)

        self.idx = 0
        self.last_save = 0
        self.full = False 


    def add(self, obs, action, reward, next_obs, done):
        np.copyto(self.obses[self.idx], obs)
        np.copyto(self.actions[self.idx], action)
        np.copyto(self.rewards[self.idx], reward)
        np.copyto(self.next_obses[self.idx], next_obs)
        np.copyto(self.not_dones[self.idx], not done)

        self.idx = (self.idx + 1) % self.capacity 
        self.full = self.full or self.idx == 0


    def sample_img_obs(self):
        idxs = np.random.randint(
            0, self.capacity if self.full else self.idx, size=self.batch_size
        )

        obses = self.obses[idxs]
        orig_obses = self.obses[idxs]
        next_obses = self.next_obses[idxs]
        pos = obses.copy()      # shallow copy (different memory location)

        obses = self.transform1(obses, self.image_size)
        obses = torch.as_tensor(obses, device=self.device).float()
        if self.transform2 is not None:
            obses = self.transform2(obses)

        orig_obses = self.transform1(orig_obses, self.image_size)
        next_obses = self.transform1(next_obses, self.image_size)
        next_obses = torch.as_tensor(next_obses, device=self.device).float()
        if self.transform2 is not None:
            next_obses = self.transform2(next_obses)

        pos = self.transform1(pos, self.image_size)
        pos = torch.as_tensor(pos, device=self.device).float()
        if self.transform2 is not None:
            pos = self.transform2(pos)


        orig_obses = torch.as_tensor(orig_obses, device=self.device).float()
        actions = torch.as_tensor(self.actions[idxs], device=self.device)
        rewards = torch.as_tensor(self.rewards[idxs], device=self.device)
        not_dones = torch.as_tensor(self.not_dones[idxs], device=self.device)

        info_dict = dict(obs_anchor=obses, obs_pos=pos, time_anchor=None, time_pos=None)

        return orig_obses, obses, actions, rewards, next_obses, not_dones, info_dict 


    def __getitem__(self, index):

        ## error here ... check

        idx = np.random.randint(0, self.capacity if self.full else self.idx, size=1)
        idx = idx[0]
        obs = self.obses[idx]
        action = self.actions[idx]
        reward = self.rewards[idx]
        next_obs = self.next_obses[idx]
        not_done = self.not_dones[idx]        

        return obs, action, reward, next_obs, not_done


    def __len__(self):
        return self.capacity if self.full else self.idx


class Config:
    def __init__(self, 
                 env,
                 replay_buffer,
                 train,
                 eval,
                 critic,
                 actor,
                 encoder,
                 decoder,
                 predictor,
                 sac,
                 params) -> None:
        
        self.env = env 
        self.replay_buffer = replay_buffer
        self.eval = eval 
        self.train = train 
        self.critic = critic
        self.actor = actor 
        self.encoder = encoder 
        self.decoder = decoder 
        self.predictor = predictor
        self.sac = sac 
        self.params = params 


    @classmethod
    def from_json(cls, cfg):
        params = json.loads(json.dumps(cfg), object_hook=HelperObject)
        return cls(
            params.env,
            params.replay_buffer,
            params.train,
            params.eval,
            params.critic,
            params.actor,
            params.encoder,
            params.decoder,
            params.predictor,
            params.sac,
            params.params
        )

class HelperObject(object):
    """ Helper class to convert json to Python Object"""
    def __init__(self, dict_) -> None:
        self.__dict__.update(dict_)