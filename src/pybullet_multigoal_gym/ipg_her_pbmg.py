'''
IPG + HER
Originally contributed by Mr. Hayden Sampson
URL: https://github.com/hayden750/DeepHEC

Input frames can be stacked together.
'''

import sys
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import os
import datetime
import random
import sys
from collections import deque
import wandb

# Add the current folder to python's import path
current_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(current_dir)
sys.path.append(os.path.dirname(current_dir))

# Local imports
from FeatureNet import FeatureNetwork, CNNLSTMFeatureNetwork
from buffer import HERBuffer
from utils import uniquify
from ipg_her import IPGHERAgent
import pybullet_multigoal_gym as pmg

class IPGHERAgent_pbmg(IPGHERAgent):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    # Validation routine
    def validate(self, max_eps=50):
        ep_reward_list = []
        for ep in range(max_eps):

            # empty the buffer for each episode 
            if self.stack_size > 1:
                state_buffer = []
                goal_buffer = []

            # initial state
            if self.image_input:
                goal_obs = np.asarray(self.env.reset()['desired_goal_img'], dtype=np.float32) / 255.0
                state_obs = np.asarray(self.env.reset()['observation'], dtype=np.float32) / 255.0
            else:
                goal_obs = self.env.reset()['desired_goal']
                state_obs = self.env.reset()['observation']


            t = 0
            ep_reward = 0
            while True:
                if self.stack_size > 1:
                    state_buffer.append(state_obs)
                    goal_buffer.append(goal_obs)
                    state = self.prepare_input(state_buffer)
                    goal = self.prepare_input(goal_buffer)
                else:
                    state = state_obs
                    goal = goal_obs

                action = self.policy(state, goal, deterministic=True)
                next_obs, reward, done, _ = self.env.step(action)

                # convert into positive reward for this environment
                reward = 1 if reward == 0 else 0  
                if self.image_input:
                    next_state_obs = np.asarray(next_obs['observation'], dtype=np.float32) / 255.0
                else:
                    next_state_obs = next_obs['observation']
                    
                    
                # convert negative reward to positive reward
                reward = 1 if reward == 0 else 0
                state_obs = next_state_obs
                ep_reward += reward
                t += 1
                if done:
                    ep_reward_list.append(ep_reward)
                    break
        # outside for loop
        mean_ep_reward = np.mean(ep_reward_list)
        return mean_ep_reward

    
    def run(self):

        if self.filename is not None:
            self.filename = uniquify(self.path + self.filename)


        # initial state
        if self.image_input:
            goal_obs = np.asarray(self.env.reset()['desired_goal_img'], dtype=np.float32) / 255.0
            state_obs = np.asarray(self.env.reset()['observation'], dtype=np.float32) / 255.0
        else:
            state_obs = self.env.reset()['observation']
            goal_obs = self.env.reset()['desired_goal']

    
        if self.stack_size > 1:
            state_buffer = []
            goal_buffer = []
            next_state_buffer = []


        start = datetime.datetime.now()
        val_scores = []       # validation scores
        best_score = -np.inf
        s_scores = []               # all season scores
        ep_lens = []                # episode lengths 
        ep_scores = []              # episodic rewards
        self.episodes = 0           # global episode count
        for s in range(self.SEASONS):
            # discard trajectories from previous season
            states, next_states, actions, rewards, dones, goals = [], [], [], [], [], []
            ep_experience = []     # episodic experience buffer
            s_score = 0         # season score
            ep_score = 0        # episodic reward
            ep_cnt = 0          # episodes in each season
            ep_len = 0          # length of each episode
            done = False
            for _ in range(self.training_batch):    # time steps

                if self.stack_size > 1:
                    state_buffer.append(state_obs)
                    goal_buffer.append(goal_obs)
                    state = self.prepare_input(state_buffer)
                    goal = self.prepare_input(goal_buffer)
                else:
                    state = state_obs
                    goal = goal_obs

                # Take an action according to its current policy
                action = self.policy(state, goal)

                # obtain reward from the environment
                next_obs, reward, done, _ = self.env.step(action)

                # make negative reward to positive
                reward = 1 if reward == 0 else 0

                if self.image_input: 
                    next_state_obs = np.asarray(next_obs['observation'], dtype=np.float32) / 255.0
                    achieved_goal = np.asarray(next_obs['achieved_goal_img'], dtype=np.float32) / 255.0
                else:
                    next_state_obs = next_obs['observation']
                    achieved_goal = next_obs['achieved_goal']
                
                # stacking 
                if self.stack_size > 1:
                    next_state_buffer.append(next_state_obs)
                    next_state = self.prepare_input(next_state_buffer)
                else:
                    next_state = next_state_obs 
                

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
                ep_score += reward
                ep_len += 1

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

                    if self.WB_LOG:
                        wandb.log({
                            'Episodes' : self.episodes, 
                            'mean_ep_score': np.mean(ep_scores),
                            'mean_ep_len' : np.mean(ep_lens)})
                        if self.episodes % 500 == 0 and self.image_input:
                            obsv_img = wandb.Image(state_obs)
                            wandb.log({'obsvn_img': obsv_img})

                    # prepare for next episode
                    if self.image_input:
                        goal_obs = np.asarray(self.env.reset()['desired_goal_img'], dtype=np.float32) / 255.0
                        state_obs = np.asarray(self.env.reset()['observation'], dtype=np.float32) / 255.0
                    else:
                        state_obs = self.env.reset()['observation']
                        goal_obs = self.env.reset()['desired_goal']

                    ep_len, ep_score = 0, 0
                    done = False

                    if self.stack_size > 1:
                        state_buffer = []
                        next_state_buffer = []
                        goal_buffer = []

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
            mean_ep_len = np.mean(ep_lens)

            # validation
            val_score = self.validate()
            val_scores.append(val_score)
            mean_val_score = np.mean(val_scores)

            if mean_s_score > best_score:
                best_model_path = self.path + 'best_model/'
                self.save_model(best_model_path)
                print('Season: {}, Update best score: {}-->{}, Model saved!'.format(s, best_score, mean_s_score))
                best_score = mean_s_score
                print('Season: {}, Validation Score: {}, Mean Validation Score: {}'\
                    .format(s, val_score, mean_val_score))

            if self.WB_LOG:
                wandb.log({'Season Score' : s_score, 
                            'Mean Season Score' : mean_s_score,
                            'Actor Loss' : actor_loss,
                            'Critic Loss' :critic_loss,
                            'Mean episode length' : mean_ep_len,
                            'val_score': val_score, 
                            'mean_val_score': mean_val_score,
                            'ep_per_season' : ep_cnt, 
                            'Season' : s})

            if self.chkpt_freq is not None and s % self.chkpt_freq == 0:          
                chkpt_path = self.path + 'chkpt_{}/'.format(s)
                self.save_model(chkpt_path)

            if self.filename is not None:
                with open(self.filename, 'a') as file:
                    file.write('{}\t{}\t{}\t{:.2f}\t{:.2f}\t{:.2f}\t{:.2f}\t{:.2f}\t{:.2f}\t{:.2f}\n'
                            .format(s, self.episodes, ep_cnt, mean_ep_len,
                                    s_score, mean_s_score, actor_loss, critic_loss,
                                    val_score, mean_val_score))

            if self.success_value is not None:
                if best_score > self.success_value:
                    print('Problem is solved in {} episodes with score {}'.format(self.episodes, best_score))
                    break

        # end of season loop
        end = datetime.datetime.now()
        print('Time to completion: {}'.format(end-start))
        print('Mean episodic score over {} episodes: {:.2f}'.format(self.episodes, np.mean(ep_scores)))
        self.env.close()

        # Save the final model
        final_model_path = self.path + 'final_model/'
        self.save_model(final_model_path)
