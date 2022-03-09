# Implementing CURL-SAC Algorithm
import numpy as np
import tensorflow as tf

from common.siamese_network import Encoder, SiameseNetwork
from algo.sac import SACActor, SACAgent, SACCritic, SACActor

class curlSacAgent(SACAgent):
    def __init__(self, state_size, action_size, feature_dim, curl_latent_dim, action_upper_bound, buffer_capacity=100000, batch_size=128, epochs=50, learning_rate=0.0003, alpha=0.2, gamma=0.99, polyak=0.995, use_attention=None, filename=None, wb_log=False, path='./'):
        super().__init__(state_size, action_size, action_upper_bound, buffer_capacity, batch_size, epochs, learning_rate, alpha, gamma, polyak, use_attention, filename, wb_log, path)
        
        assert state_size.ndim == 3, 'state_size must be a 3D tensor'

        self.feature_dim = feature_dim  # encoder feature dim
        self.curl_latent_dim = curl_latent_dim  
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

    def train(self):
        super().train()
        self.actor.train()
        self.critic1.train()



    def run(self):
        pass


    def validate(self, env, max_eps=50):
        ep_reward_list = []
        for ep in range(max_eps):
            obs = env.reset()

            if self.image_input:
                state = np.asarray(obs['observation'], dtype=np.float32) / 255.0
                goal = np.asarray(obs['desired_goal_img'], dtype=np.float32) / 255.0
            else:
                state = obs['observation']
                goal = obs['desired_goal']

            t = 0
            ep_reward = 0
            while True:
                action, _ = self.sample_action(state, goal)
                next_obs, reward, done, _ = env.step(action)

                if self.image_input:
                    next_state = np.asarray(next_obs['observation'], dtype=np.float32) / 255.0
                    next_goal = np.asarray(next_obs['desired_goal_img'], dtype=np.float32) / 255.0
                else:
                    next_state = next_obs['observation']
                    next_goal = next_obs['desired_goal']

                # convert negative reward to positive reward
                reward = 1 if reward == 0 else 0
                state = next_state
                goal = next_goal
                ep_reward += reward
                t += 1
                if done:
                    ep_reward_list.append(ep_reward)
                    break
        # outside for loop
        mean_ep_reward = np.mean(ep_reward_list)
        return mean_ep_reward


