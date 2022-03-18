# Implementing CURL-SAC Algorithm
import numpy as np
import tensorflow as tf
from torch import rand

from common.siamese_network import Encoder, SiameseNetwork
from algo.sac import SACActor, SACAgent, SACCritic, SACActor
from common.buffer import Buffer
from common.utils import random_crop, uniquify

class curlSacAgent(SACAgent):
    def __init__(self, state_size, 
                action_size, 
                feature_dim, 
                curl_latent_dim, 
                action_upper_bound, 
                buffer_capacity=100000, 
                batch_size=128, epochs=50, 
                learning_rate=0.0003, alpha=0.2, 
                gamma=0.99, polyak=0.995, 
                cropped_img_size=50,
                use_attention=None, filename=None, wb_log=False, path='./'):
        super().__init__(state_size, action_size, action_upper_bound, 
            buffer_capacity, batch_size, epochs, learning_rate, 
            alpha, gamma, polyak, use_attention, 
            filename, wb_log, path)
        
        assert state_size.ndim == 3, 'state_size must be a 3D tensor'

        self.feature_dim = feature_dim  # encoder feature dim
        self.curl_latent_dim = curl_latent_dim  
        self.cropped_img_size = cropped_img_size # needed for contrastive learning

        # create the encoder
        self.encoder = Encoder(self.state_size, self.feature_dim)

        # Contrastive network
        cont_net = SiameseNetwork(self.state_size, self.curl_latent_dim, self.feature_dim)

        # Actor
        self.actor = SACActor(state_size, action_size, action_upper_bound, encoder=self.encoder)

        # create two critics
        self.critic1 = SACCritic(self.state_size, self.action_size,
                                 self.lr, self.gamma, self.feature)
        self.critic2 = SACCritic(self.state_size, self.action_size,
                                 self.lr, self.gamma, self.encoder)

        # create two target critics
        self.target_critic1 = SACCritic(self.state_size, self.action_size,
                                 self.lr, self.gamma, self.encoder)
        self.target_critic2 = SACCritic(self.state_size, self.action_size,
                                 self.lr, self.gamma, self.encoder)


    # train the curl agent
    def train(self): 
        super().train()
        self.actor.train()
        self.critic1.train()

    def create_image_pairs(self):

        # sample a minibatch from the replay buffer
        states, actions, rewards, next_states, dones = self.buffer.sample()

        obs_a = random_crop(states, self.cropped_img_size)  # anchor images
        obs_p = random_crop(states, self.cropped_img_size)  # positive images
        obs_n = random_crop(next_states, self.cropped_img_size)  # negative images


    # main training loop
    def run(self, env, max_seasons=200,
            training_batch=2560,
            WB_LOG=False, success_value=None,
            filename=None, chkpt_freq=None,
            path='./'):

        if filename is not None:
            filename = uniquify(path + filename)

        

        




