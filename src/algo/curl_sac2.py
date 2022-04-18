# Implementing CURL-SAC Algorithm
import numpy as np
import tensorflow as tf
import gym
from collections import deque
import wandb

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
                stack_size=3,
                use_attention=None, filename=None, wb_log=False, path='./'):
        super().__init__(state_size, action_size, action_upper_bound, 
            buffer_capacity, batch_size, epochs, learning_rate, 
            alpha, gamma, polyak, use_attention, 
            filename, wb_log, path)
        
        assert state_size.ndim == 3, 'state_size must be a 3D tensor'

        self.feature_dim = feature_dim  # encoder feature dim
        self.curl_latent_dim = curl_latent_dim  
        self.cropped_img_size = cropped_img_size # needed for contrastive learning
        self.stack_size = stack_size
        
        # final observation shape obtained after augmentation
        self.obs_shape = (self.cropped_img_size, self.cropped_img_size, self.state_size[2])

        # create the encoder
        self.encoder = Encoder(self.obs_shape, self.feature_dim)

        # Contrastive network
        cont_net = SiameseNetwork(self.obs_shape, self.curl_latent_dim, self.feature_dim)

        # Actor
        self.actor = SACActor(self.obs_shape, action_size, action_upper_bound, encoder=self.encoder)

        # create two critics
        self.critic1 = SACCritic(self.obs_shape, self.action_size,
                                 self.lr, self.gamma, self.feature)
        self.critic2 = SACCritic(self.obs_shape, self.action_size,
                                 self.lr, self.gamma, self.encoder)

        # create two target critics
        self.target_critic1 = SACCritic(self.obs_shape, self.action_size,
                                 self.lr, self.gamma, self.encoder)
        self.target_critic2 = SACCritic(self.obs_shape, self.action_size,
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
        

    
    def validate(self, env, max_eps=50):
        pass

    # main training loop
    def run(self, env, max_training_steps=100000,
            eval_freq=1000, init_steps=1000,
            WB_LOG=False): 

        episode, ep_reward, done = 0, 0, True
        ep_rewards = []
        for step in range(max_training_steps):

            if step % eval_freq == 0:
                self.validate(env, max_eps=50)


            if done:
                done = False
                ep_reward = 0
                episode += 1 
                ep_step = 0
                obs = env.reset()
                

            # make an observation                
            state = np.asarray(obs['observation'], dtype=np.float32) / 255.0

            # take an action
            action = self.sample_action(state)

            # obtain rewards
            next_obs, reward, done, _ = self.env.step(action)
            next_state = np.asarray(next_obs['observation'], dtype=np.float32)

            # record experience
            self.buffer.record([state, action, reward, next_state, done])

            # train
            if step > init_steps:
                self.train()

            ep_reward += reward 
            ep_rewards.append(reward)

            if WB_LOG:
                wandb.log({
                    'episodes' : episode,
                    'ep_reward' : ep_reward,
                    'mean_ep_reward' : np.mean(ep_rewards),
                })

            state = next_state    
            ep_step += 1

        env.close()
        print('Mean episodic score over {} episodes: {:.2f}'\
                            .format(episode, np.mean(ep_rewards)))
                            


            


        

        




