"""
SAC + HER Algorithm applied to pybullet_multi_goal environment.


"""
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.keras import layers
import os
import datetime
from collections import deque
import wandb
import sys

# add current directory to python module path
current_directory = os.path.dirname(os.path.realpath(__file__))
sys.path.append(current_directory)
sys.path.append(os.path.dirname(current_directory))

# Local imports
from algo.sac_her import SACHERAgent
from common.FeatureNet import FeatureNetwork
from common.buffer import HERBuffer
from common.utils import uniquify


class SACHERAgent_pbmg(SACHERAgent):

    def add_her_experience(self, ep_experience, hind_goal):
        for i in range(len(ep_experience)):
            state_ = ep_experience[i][0]
            action_ = ep_experience[i][1]
            next_state_ = ep_experience[i][3]
            goal_ = ep_experience[i][5]

            done_ = np.array_equal(goal_, hind_goal)
            hind_reward = 0 if done_ else -1

            self.buffer.record(
                [state_, action_, hind_reward, next_state_, done_, hind_goal]
            )

    def validate(self, env, max_eps=20):
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
                else:
                    next_state = next_obs['observation']

                state = next_state
                ep_reward += reward
                t += 1
                if done:
                    ep_reward_list.append(ep_reward)
                    break
        # outside for loop
        mean_ep_reward = np.mean(ep_reward_list)
        return mean_ep_reward

    def run(self, env, max_episodes=5000, train_freq=1, WB_LOG=True):


        start = datetime.datetime.now()
        
        val_scores = []                 # validation scores 
        best_score = -np.inf
        ep_lens = []        # episodic length
        ep_scores = []      # All episodic scores
        ep_actor_losses = []    # actor losses
        ep_critic_losses = []   # critic losses
        ep_alpha_losses = []
        global_steps = 0

        for ep in range(max_episodes):
            # initial state
            obs = env.reset()
            if self.image_input:
                state = np.asarray(obs['observation'], dtype=np.float32) / 255.0
                goal = np.asarray(obs['desired_goal_img'], dtype=np.float32) / 255.0
            else:
                state = obs['observation']
                goal = obs['desired_goal']

            ep_experience = []    # required for HER
            ep_len = 0          # episode length
            ep_score = 0        # score for each episode
            done = False
            while not done:
                action, _ = self.sample_action(state, goal)
                next_obs, reward, done, _ = env.step(action)

                if self.image_input:
                    next_state = np.asarray(next_obs['observation'], dtype=np.float32) / 255.0
                    achieved_goal = np.asarray(next_obs['achieved_goal_img'], dtype=np.float32) / 255.0
                else:
                    next_state = next_obs['observation']
                    achieved_goal = next_obs['achieved_goal']

                # store in replay buffer for off-policy training
                self.buffer.record([state, action, reward, next_state, done, goal])

                # also store experience in temporary buffer
                ep_experience.append([state, action, reward, next_state, done, goal])

                state = next_state
                ep_score += reward
                ep_len += 1     # no. of time steps in each episode
                global_steps += 1
            # end of episode
                
            ep_scores.append(ep_score)
            ep_lens.append(ep_len) 
            mean_ep_score = np.mean(ep_scores)
            
            # HER strategies
            hind_goal = achieved_goal

            self.add_her_experience(ep_experience, hind_goal) 
            ep_experience = [] # clear temporary buffer

            if ep % train_freq == 0:
                # train
                a_loss, c_loss, alpha_loss = self.train()

                ep_actor_losses.append(a_loss)
                ep_critic_losses.append(c_loss)
                ep_alpha_losses.append(alpha_loss)

                # validate
                val_score = self.validate(env, max_eps=20)
                val_scores.append(val_score)

                if WB_LOG:
                    wandb.log({
                        'episodes' : ep,
                        'mean_ep_score': np.mean(ep_scores),
                        'mean_ep_len' : np.mean(ep_lens),
                        'mean_val_score' : np.mean(val_scores),
                        'ep_actor_loss' : np.mean(ep_actor_losses),
                        'ep_critic_loss' : np.mean(ep_critic_losses),
                        'ep_alpha_loss' : np.mean(ep_alpha_losses)})


            if mean_ep_score > best_score:
                best_model_path =  './log/best_model/'
                self.save_model(best_model_path)
                best_score = mean_ep_score
                print('Episode: {}, Update best score: {:.3f}-->{:.3f}, Model saved!'.format(ep, best_score, mean_ep_score))
                print('Episode: {}, Validation Score: {:.3f}, Mean Validation Score: {}' \
                .format(ep, val_score, np.mean(val_scores)))

        # end offor-loop
        end = datetime.datetime.now()
        print('Time to Completion: {}'.format(end - start))
        env.close()
        print('Mean episodic score over {} episodes: {:.2f}'.format(ep, mean_ep_score))
        print('Total number of steps =', global_steps)
