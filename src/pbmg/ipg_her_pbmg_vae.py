'''
IPG + HER + VAE
Originally contributed by Mr. Hayden Sampson
URL: https://github.com/hayden750/DeepHEC

Input frames can be stacked together.

20/01/2022: Added stacking frames.
'''

import sys
import numpy as np
from numpy.lib.npyio import load
import tensorflow as tf
import tensorflow_probability as tfp
import os
import datetime
import random
import sys
from collections import deque
import wandb
import imageio 
import matplotlib.pyplot as plt

# Add the current folder to python's import path
current_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(current_dir)
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
sys.path.append(os.path.join(parent_dir, 'common/'))

# Local imports
from common.utils import uniquify, prepare_stacked_images
from algo.ipg_her import IPGHERAgent, IPGHERActor, DDPGCritic, Baseline
from common.VariationAutoEncoder import Encoder

class IPGHERAgent_pbmg(IPGHERAgent):
    def __init__(self, state_size, action_size, upper_bound, buffer_capacity=100000, batch_size=256, learning_rate=0.0002, epochs=20, epsilon=0.2, gamma=0.95, lmbda=0.7, use_attention=None, use_her=None, stack_size=0, use_lstm=None):
        super().__init__(state_size, action_size, upper_bound, buffer_capacity=buffer_capacity, batch_size=batch_size, learning_rate=learning_rate, epochs=epochs, epsilon=epsilon, gamma=gamma, lmbda=lmbda, use_attention=use_attention, use_her=use_her, stack_size=stack_size, use_lstm=use_lstm)

        if self.image_input:
            if self.stack_size > 1:
                h, w, c = state_size
                self.state_size = (h, w, c * self.stack_size)
                self.goal_size = self.state_size
            self.feature = Encoder(self.state_size, latent_dim=10) # Use encoder to get latent representation
        else:
            self.feature = None

        # Actor Model
        self.actor = IPGHERActor(state_size=self.state_size, goal_size=self.goal_size,
                              action_size=self.action_size, upper_bound=self.upper_bound,
                              lr=self.lr, epsilon=self.epsilon, feature=self.feature)
        # Critic Model
        self.critic = DDPGCritic(state_size=self.state_size, action_size=self.action_size,
                                 learning_rate=self.lr, gamma=self.gamma, feature_model=self.feature)
        # Baseline Model
        self.baseline = Baseline(state_size=self.state_size, action_size=self.action_size,
                                lr=self.lr, feature=self.feature)


    # Validation routine
    def validate(self, env, max_eps=50, render=False, load_path=None):

        if load_path is not None:
            self.load_model(load_path)

        if render:
            images = []
        ep_reward_list = []
        for ep in range(max_eps):

            # obtain observation
            obs = env.reset()

            # empty the buffer for each episode 
            if self.stack_size > 1:
                state_buffer = []
                goal_buffer = []

            # initial state
            if self.image_input:
                goal_obs = np.asarray(obs['desired_goal_img'], dtype=np.float32) / 255.0
                state_obs = np.asarray(obs['observation'], dtype=np.float32) / 255.0
            else:
                goal_obs = env.reset()['desired_goal']
                state_obs = env.reset()['observation']

            if render:
                img = env.render(mode='rgb_array')
                images.append(img)

            ep_reward = 0
            while True:

                if self.stack_size > 1:
                    state_buffer.append(state_obs)
                    goal_buffer.append(goal_obs)
                    state = prepare_stacked_images(state_buffer)
                    goal = prepare_stacked_images(goal_buffer)
                else:
                    state = state_obs
                    goal = goal_obs

                #action = self.policy(state, goal, deterministic=True)
                action = self.policy(state, goal, deterministic=True)
                next_obs, reward, done, _ = env.step(action)


                if render:
                    img = env.render(mode='rgb_array')
                    images.append(img)

                if self.image_input:
                    next_state_obs = np.asarray(next_obs['observation'], dtype=np.float32) / 255.0
                    next_goal_obs = np.asarray(next_obs['desired_goal_img'], dtype=np.float32) / 255.0
                else:
                    next_state_obs = next_obs['observation']
                    next_goal_obs = next_obs['desired_goal']
                    
                    
                # convert negative reward to positive reward
                reward = 1 if reward == 0 else 0
                state_obs = next_state_obs
                goal_obs = next_goal_obs
                ep_reward += reward
                if done:
                    ep_reward_list.append(ep_reward)
                    break
        # outside for loop
        mean_ep_reward = np.mean(ep_reward_list)

        if render:
            imageio.mimsave('./pbmg_reach.gif', np.array(images), fps=10)

        return mean_ep_reward

    
    def run(self, env, max_seasons=200, training_batch=5120, 
                WB_LOG=False, success_value=None, 
                filename=None, chkpt_freq=None, path='./'):

        if filename is not None:
            filename = uniquify(path + filename)

        # initial state
        obs = env.reset()
        if self.image_input:
            goal_obs = np.asarray(obs['desired_goal_img'], dtype=np.float32) / 255.0
            state_obs = np.asarray(obs['observation'], dtype=np.float32) / 255.0
        else:
            state_obs = obs['observation']
            goal_obs = obs['desired_goal']

        if self.stack_size > 1:
            state_buffer = []
            goal_buffer = []
            next_state_buffer = []
            achieved_goal_buffer = [] # empty after each episode

        start = datetime.datetime.now()
        val_scores = []       # validation scores
        best_score = -np.inf
        s_scores = []               # all season scores
        ep_lens = []                # episode lengths 
        ep_scores = []              # episodic rewards
        self.episodes = 0           # global episode count
        self.global_time_steps = 0       # global time steps
        for s in range(max_seasons):
            # discard trajectories from previous season
            states, next_states, actions, rewards, dones, goals = [], [], [], [], [], []
            ep_experience = []     # episodic experience buffer
            s_score = 0         # season score
            ep_score = 0        # episodic reward
            ep_cnt = 0          # episodes in each season
            ep_len = 0          # length of each episode
            done = False
            for _ in range(training_batch):    # time steps

                if self.stack_size > 1:
                    state_buffer.append(state_obs)
                    goal_buffer.append(goal_obs)
                    state = prepare_stacked_images(state_buffer)
                    goal = prepare_stacked_images(goal_buffer)
                else:
                    state = state_obs
                    goal = goal_obs

                # Take an action according to its current policy
                action = self.policy(state, goal)

                # obtain reward from the environment
                next_obs, reward, done, _ = env.step(action)

                # make negative reward to positive
                reward = 1 if reward == 0 else 0

                if self.image_input: 
                    next_state_obs = np.asarray(next_obs['observation'], dtype=np.float32) / 255.0
                    achieved_goal = np.asarray(next_obs['achieved_goal_img'], dtype=np.float32) / 255.0
                    next_goal_obs = np.asarray(next_obs['desired_goal_img'], dtype=np.float32) / 255.0
                else:
                    next_state_obs = next_obs['observation']
                    achieved_goal_obs = next_obs['achieved_goal']
                    next_goal_obs = np.asarray(next_obs['desired_goal'])
                
                # stacking 
                if self.stack_size > 1:
                    next_state_buffer.append(next_state_obs)
                    achieved_goal_buffer.append(achieved_goal_obs)
                    next_state = prepare_stacked_images(next_state_buffer, self.stack_size)
                    achieved_goal = prepare_stacked_images(achieved_goal_buffer, self.stack_size)
                else:
                    next_state = next_state_obs 
                    achieved_goal = achieved_goal_obs
                

                # this is used for on-policy training
                states.append(state)
                next_states.append(next_state)
                actions.append(action)
                rewards.append(reward)
                dones.append(done)
                goals.append(goal)

                # store in replay buffer for off-policy training
                self.buffer.record([state, action, reward, next_state, done, goal])
                # Also store in a separate buffer
                ep_experience.append([state, action, reward, next_state, done, goal])

                state_obs = next_state_obs
                goal_obs = next_goal_obs
                ep_score += reward
                ep_len += 1
                self.global_time_steps += 1

                if done:
                    # HER: Final state strategy

                    hind_goal = achieved_goal

                    # Add hindsight experience to the buffer
                    # in this case, use last state as the goal_state
                    # here ep_experience buffer is cleared at the end of each episode.
                    self.add_her_experience(ep_experience, hind_goal, 
                                            self.use_her['extract_feature'])
                    # clear the local experience buffer
                    ep_experience = []

                    self.episodes += 1
                    s_score += ep_score        # season score
                    ep_cnt += 1             # episode count in a season
                    ep_scores.append(ep_score)
                    ep_lens.append(ep_len)

                    if WB_LOG:
                        wandb.log({
                            'Episodes' : self.episodes, 
                            'global_time_steps': self.global_time_steps,
                            'mean_ep_score': np.mean(ep_scores),
                            'mean_ep_len' : np.mean(ep_lens)})
                        # if self.episodes % 500 == 0 and self.image_input:
                        #     obsv_img = wandb.Image(state_obs)
                        #     wandb.log({'obsvn_img': obsv_img})

                    # prepare for next episode
                    obs = env.reset()
                    if self.image_input:
                        goal_obs = np.asarray(obs['desired_goal_img'], dtype=np.float32) / 255.0
                        state_obs = np.asarray(obs['observation'], dtype=np.float32) / 255.0
                    else:
                        state_obs = env.reset()['observation']
                        goal_obs = env.reset()['desired_goal']

                    ep_len, ep_score = 0, 0
                    done = False

                    if self.stack_size > 1:
                        state_buffer = []
                        next_state_buffer = []
                        goal_buffer = []
                        achieved_goal_buffer = []

                # end of done block
            # end of for training_batch loop

            # Add hindsight experience to the buffer
            # here we are using random goal states
            # self.add_her_experience(ep_experience)
            # clear the local experience buffer
            # ep_experience = []          # not required as we are clearing it for every season.

            # on-policy & off-policy training
            actor_loss, critic_loss = self.train(states, actions, rewards, next_states, dones, goals)

            s_score = np.mean(ep_scores[-ep_cnt : ])
            s_scores.append(s_score)
            mean_s_score = np.mean(s_scores)

            # validation
            val_score = self.validate(env)
            val_scores.append(val_score)
            mean_val_score = np.mean(val_scores)

            if mean_s_score > best_score:
                best_model_path = path + 'best_model/'
                self.save_model(best_model_path)
                print('Season: {}, Update best score: {}-->{}, Model saved!'.format(s, best_score, mean_s_score))
                best_score = mean_s_score
                print('Season: {}, Validation Score: {}, Mean Validation Score: {}'\
                    .format(s, val_score, mean_val_score))

            if WB_LOG:
                wandb.log({'Season Score' : s_score, 
                            'Mean Season Score' : mean_s_score,
                            'Actor Loss' : actor_loss,
                            'Critic Loss' :critic_loss,
                            'val_score': val_score, 
                            'mean_val_score': mean_val_score,
                            'ep_per_season' : ep_cnt, 
                            'Season' : s})

            if chkpt_freq is not None and s % self.chkpt_freq == 0:          
                chkpt_path = path + 'chkpt_{}/'.format(s)
                self.save_model(chkpt_path)

            if filename is not None:
                with open(self.filename, 'a') as file:
                    file.write('{}\t{}\t{}\t{:.2f}\t{:.2f}\t{:.2f}\t{:.2f}\t{:.2f}\t{:.2f}\t{:.2f}\n'
                            .format(s, self.episodes, ep_cnt, mean_ep_len,
                                    s_score, mean_s_score, actor_loss, critic_loss,
                                    val_score, mean_val_score))

            if success_value is not None:
                if best_score > self.success_value:
                    print('Problem is solved in {} episodes with score {}'.format(self.episodes, best_score))
                    break

        # end of season loop
        end = datetime.datetime.now()
        print('Time to completion: {}'.format(end-start))
        print('Mean episodic score over {} episodes: {:.2f}'.format(self.episodes, np.mean(ep_scores)))
        env.close()

        # Save the final model
        final_model_path = path + 'final_model/'
        self.save_model(final_model_path)

        