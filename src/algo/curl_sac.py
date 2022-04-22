# Implementing CURL-SAC Algorithm
from turtle import st
import numpy as np
import tensorflow as tf
import gym
from collections import deque
import wandb

from common.siamese_network import Encoder, SiameseNetwork
from algo.sac import SACActor, SACAgent, SACCritic, SACActor
from common.buffer import Buffer
from common.utils import center_crop_image, random_crop, uniquify


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
        
        assert len(state_size) == 3, 'state_size must be a 3D tensor'

        self.feature_dim = feature_dim  # encoder feature dim
        self.curl_latent_dim = curl_latent_dim  
        self.cropped_img_size = cropped_img_size # needed for contrastive learning
        self.stack_size = stack_size
        
        # final observation shape obtained after augmentation
        self.obs_shape = (self.cropped_img_size, self.cropped_img_size, self.state_size[2])


        # Contrastive network
        self.cont_net = SiameseNetwork(self.obs_shape, self.curl_latent_dim, self.feature_dim)

        # create the encoder
        self.actor_encoder = self.cont_net.key_encoder
        self.critic_encoder = self.cont_net.query_encoder

        # Actor
        self.actor = SACActor(self.obs_shape, action_size, 
                        action_upper_bound, self.lr, self.actor_encoder)

        # create two critics
        self.critic1 = SACCritic(self.obs_shape, self.action_size,
                                 self.lr, self.gamma, self.critic_encoder)
        self.critic2 = SACCritic(self.obs_shape, self.action_size,
                                 self.lr, self.gamma, self.critic_encoder)

        # create two target critics
        self.target_critic1 = SACCritic(self.obs_shape, self.action_size,
                                 self.lr, self.gamma, self.critic_encoder)
        self.target_critic2 = SACCritic(self.obs_shape, self.action_size,
                                 self.lr, self.gamma, self.critic_encoder)

        # alpha & buffer are defined from the parent class


    def create_image_pairs(self):
        # sample a minibatch from the replay buffer
        states, _, _, next_states, _ = self.buffer.sample()
        obs_a = random_crop(states, self.cropped_img_size)  # anchor images
        obs_p = random_crop(states, self.cropped_img_size)  # positive images
        obs_n = random_crop(next_states, self.cropped_img_size)  # negative images - not used.
        return obs_a, obs_p, obs_n 

    
    def train_encoder(self, itn_max=10):
        # train encoder using contrastive learning
        enc_losses = []
        for _ in range(itn_max):
            obs_a, obs_p, obs_n = self.create_image_pairs()
            loss = self.cont_net.train((obs_a, obs_p))
            enc_losses.append(loss)
        return np.mean(enc_losses)

    def update_target_networks(self):
        super().update_target_networks()
        self.cont_net.update_key_encoder_wts
            

    def train_actor_critic(self, itn_max=20):

        critic_losses, actor_losses, alpha_losses = [], [], []
        for _ in range(itn_max):

            # sample a minibatch from the replay buffer
            states, actions, rewards, next_states, dones = self.buffer.sample()

            states = [random_crop(states[i], self.cropped_img_size) for i in range(len(states))]
            next_states = [random_crop(states[i], self.cropped_img_size) for i in range(len(next_states))]

            # convert to tensors
            states = tf.convert_to_tensor(states, dtype=tf.float32)
            actions = tf.convert_to_tensor(actions, dtype=tf.float32)
            rewards = tf.convert_to_tensor(rewards, dtype=tf.float32)
            next_states = tf.convert_to_tensor(next_states, dtype=tf.float32)
            dones = tf.convert_to_tensor(dones, dtype=tf.float32)

            # update Q network weights
            critic_loss = self.update_q_networks(states, actions, rewards, next_states, dones)
                                    
            # update policy networks
            actor_loss = self.actor.train(states, self.alpha, self.critic1, self.critic2)

            # update entropy coefficient
            alpha_loss = self.update_alpha(states)

            # update target network weights
            self.update_target_networks()

            critic_losses.append(critic_loss)
            actor_losses.append(actor_loss)
            alpha_losses.append(alpha_loss)
        # itn loop ends here
        mean_critic_loss = np.mean(critic_losses)
        mean_actor_loss = np.mean(actor_losses)
        mean_alpha_loss = np.mean(alpha_losses)

        return mean_actor_loss, mean_critic_loss, mean_alpha_loss

    def validate(self, env, max_eps=20):
        '''
        evaluate model performance
        '''
        ep_reward_list = []
        for _ in range(max_eps):
            obs = env.reset()
            state = np.asarray(obs, dtype=np.float32) / 255.0
            t = 0
            ep_reward = 0
            while True:
                action, _ = self.sample_action(state)
                next_obs, reward, done, _ = env.step(action)
                next_state = np.asarray(next_obs, dtype=np.float32) / 255.0
                state = next_state 
                ep_reward += reward 
                t += 1
                if done:
                    ep_reward_list.append(ep_reward)
                    break
        # outside the loop
        mean_ep_reward = np.mean(ep_reward_list)
        return mean_ep_reward


    # main training loop
    def run(self, env, max_training_steps=100000,
            eval_freq=1000, init_steps=1000,
            ac_train_freq=5, enc_train_freq=2, 
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
            
            # actor takes cropped_img_size
            cropped_state = center_crop_image(state, out_h=self.cropped_img_size)

            # take an action
            action = self.sample_action(cropped_state)

            # obtain rewards
            next_obs, reward, done, _ = self.env.step(action)
            next_state = np.asarray(next_obs['observation'], dtype=np.float32)

            # record experience
            self.buffer.record([state, action, reward, next_state, done])

            # train
            if step > init_steps and step % ac_train_freq == 0:
                a_loss, c_loss, alpha_loss = self.train_actor_critic()

            if step > init_steps and step % enc_train_freq == 0:
                enc_loss = self.train_encoder()

            ep_reward += reward 
            ep_rewards.append(reward)

            if WB_LOG:
                wandb.log({
                    'episodes' : episode,
                    'ep_reward' : ep_reward,
                    'mean_ep_reward' : np.mean(ep_rewards),
                    'mean_actor_loss' : a_loss,
                    'mean_critic_loss' : c_loss,
                    'mean_alpha_loss' : alpha_loss,
                    'mean_enc_loss' : enc_loss,
                })

            state = next_state    
            ep_step += 1

        env.close()
        print('Mean episodic score over {} episodes: {:.2f}'\
                            .format(episode, np.mean(ep_rewards)))



            


        

        




