"""
SAC + HER Algorithm.

We implement two strategies for selecting hind_goal

- Last state as hind goal
- Last successful state

"""
from inspect import currentframe
import numpy as np
import tensorflow as tf
from tensorflow.python.keras.backend import dtype
import tensorflow_probability as tfp
from tensorflow.keras import layers
import os
import datetime
import random
from collections import deque
import wandb
import sys

# add current directory to python module path
current_directory = os.path.dirname(os.path.realpath(__file__))
sys.path.append(current_directory)
sys.path.append(os.path.dirname(current_directory))

# Local imports
from common.FeatureNet import FeatureNetwork
from common.buffer import HERBuffer
from common.utils import uniquify
from algo.sac import SACActor, SACCritic, SACAgent


###############
# ACTOR NETWORK
###############


class SACHERAgent(SACAgent):
    def __init__(self, state_size, action_size, action_upper_bound,
                buffer_capacity=100000, batch_size=128, epochs=50, 
                learning_rate=0.0003, alpha=0.2, gamma=0.99,
                polyak=0.995, use_attention=None, 
                filename=None, wb_log=False, path='./'):

        state_size = 2 * state_size
        super().__init__(state_size, action_size, action_upper_bound,
        buffer_capacity, batch_size, epochs, learning_rate,
        alpha, gamma, polyak, use_attention, filename,
        wb_log, path='./')                

        self.buffer = HERBuffer(self.buffer_capacity, self.batch_size)
        
    def add_her_experience(self, ep_experience, hind_goal):
        for i in range(len(ep_experience)):
            if hind_goal is None:   # future state strategy
                future = np.random.randint(i, len(ep_experience))
                goal_ = ep_experience[future][3]
            else:
                goal_ = hind_goal

            state_ = ep_experience[i][0]
            action_ = ep_experience[i][1]
            next_state_ = ep_experience[i][3]

            done_, reward_ = self.her_reward_func(next_state_, goal_)

            # add new experience to the main buffer
            self.buffer.record([state_, action_, reward_, next_state_, done_, goal_])

    def her_reward_func(self, state, goal, thr=0.2):
        # input: numpy array, output: numpy value
        good_done = np.linalg.norm(state - goal) <= thr 
        reward = 1 if good_done else 0
        return good_done, reward

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
                action, _ = self.policy(state, goal)
                next_obs, reward, done, _ = self.env.step(action)

                # make reward positive
                reward = 1 if reward == 0 else 0

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

    def run(self, env, max_episodes=1000, train_freq=20):

        if self.filename is not None: 
            self.filename = uniquify(self.path + self.filename)


        start = datetime.datetime.now()
        val_scores = []                 # validation scores 
        best_score = -np.inf
        ep_lens = []        # episodic length
        ep_scores = []      # All episodic scores
        avg_ep_scores = []  # average episodic scores
        avg_actor_losses = []    # actor losses
        avg_critic_losses = []   # critic losses
        avg_alpha_losses = []

        for ep in range(max_episodes):

            obs = env.reset()
            
            if self.image_input:
                state = np.asarray(obs['observation'], dtype=np.float32) / 255.0
                goal = np.asarray(obs['desired_goal_img'], dtype=np.float32) / 255.0
            else:
                state = obs['observation']
                goal = obs['desired_goal']

            # extended state
            state_ext = np.concatenate((state, goal), axis=0)

            ep_score = 0 
            ep_len = 0
            ep_experience = []
            done = False
            while not done:
                action, _ = self.policy(state_ext)
                next_obs, reward, done, _ = self.env.step(action)

                # make reward positive
                reward = 1 if reward == 0 else 0

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
                self.global_steps += 1
            # while loop ends here
                
            ep_scores.append(ep_score)
            ep_lens.append(ep_len) 

            # HER strategies
            hind_goal = achieved_goal  # terminal goal for the current episode


            self.add_her_experience(ep_experience, hind_goal, extract_feature=False)
            ep_experience = [] # clear temporary buffer

                    # off-policy training after each episode
                    if self.episodes % train_freq == 0:
                        a_loss, c_loss, alpha_loss = self.train()

                        ep_actor_losses.append(a_loss)
                        ep_critic_losses.append(c_loss)

                        if self.WB_LOG:
                            wandb.log({
                            'ep_actor_loss' : a_loss,
                            'ep_critic_loss' : c_loss,
                            'ep_alpha_loss' : alpha_loss})

                    if self.WB_LOG:
                        wandb.log({
                            'Episodes' : self.episodes, 
                            'mean_ep_score': np.mean(ep_scores),
                            'mean_ep_len' : np.mean(ep_lens)})
                    
                    # prepare for next episode
                    state = np.asarray(self.env.reset(), dtype=np.float32) / 255.0
                    goal = np.asarray(self.env.reset(), dtype=np.float32) / 255.0
                    ep_len, ep_score = 0, 0
                    done = False
                # done block ends here
            # end of one season
            
            s_score = np.mean(ep_scores[-ep_cnt : ])
            s_scores.append(s_score)
            mean_ep_score = np.mean(ep_scores)
            mean_ep_len = np.mean(ep_lens)
            mean_s_score = np.mean(s_scores)
            mean_actor_loss = np.mean(ep_actor_losses[-ep_cnt:])
            mean_critic_loss = np.mean(ep_critic_losses[-ep_cnt:])

            # run validation once in each iteration
            val_score = self.validate()
            val_scores.append(val_score)
            mean_val_score = np.mean(val_scores)

            if mean_s_score > best_score:
                best_model_path = self.path + 'best_model/'
                os.makedirs(best_model_path, exist_ok=True)
                self.save_model(best_model_path)
                best_score = mean_s_score
                print('Season: {}, Update best score: {}-->{}, Model saved!'.format(s, best_score, mean_ep_score))
                print('Season: {}, Validation Score: {}, Mean Validation Score: {}' \
                .format(s, val_score, mean_val_score))

            if self.WB_LOG:
                wandb.log({'Season Score' : s_score, 
                            'Mean Season Score' : mean_s_score,
                            'Actor Loss' : mean_actor_loss,
                            'Critic Loss' : mean_critic_loss,
                            'Mean episode length' : mean_ep_len,
                            'val_score': val_score, 
                            'mean_val_score': mean_val_score,
                            'ep_per_season' : ep_cnt,
                            'Season' : s})

            if self.chkpt_freq is not None and s % self.chkpt_freq == 0:
                chkpt_path = self.path + 'chkpt_{}/'.format(s)
                os.makedirs(chkpt_path, exist_ok=True)
                self.save_model(chkpt_path)

            if self.filename is not None:
                with open(self.filename, 'a') as file:
                    file.write('{}\t{}\t{}\t{:.2f}\t{:.2f}\t{:.2f}\t{:.2f}\t{:.2f}\t{:.2f}\t{:.2f}\t{:.2f}\n'
                            .format(s, self.episodes, ep_cnt, mean_ep_len,
                                    s_score, mean_s_score, mean_actor_loss, mean_critic_loss, alpha_loss,
                                    val_score, mean_val_score))

            if self.success_value is not None:
                if best_score > self.success_value:
                    print('Problem is solved in {} episodes with score {}'.format(s, best_score))
                    print('Mean Episodic score: {}'.format(mean_s_score))
                    break

        # end of season-loop
        end = datetime.datetime.now()
        print('Time to Completion: {}'.format(end - start))
        self.env.close()
        print('Mean episodic score over {} episodes: {:.2f}'.format(self.episodes, np.mean(ep_scores)))

    def her_reward_func_1(self, state, goal, thr=0.3):
        tf_state = tf.expand_dims(tf.convert_to_tensor(state, dtype=tf.float32), axis=0)
        tf_goal = tf.expand_dims(tf.convert_to_tensor(goal, dtype=tf.float32), axis=0)
        
        state_feature = tf.squeeze(self.feature(tf_state))
        goal_feature = tf.squeeze(self.feature(tf_goal))

        good_done = tf.linalg.norm(state_feature - goal_feature) <= thr
        reward = 1 if good_done else 0
        return good_done, reward

    def her_reward_func_2(self, state, goal, thr=0.2):
        # input: numpy array, output: numpy value
        good_done = np.linalg.norm(state - goal) <= thr 
        reward = 1 if good_done else 0
        return good_done, reward

    def add_her_experience(self, ep_experience, hind_goal, extract_feature=False):
        for i in range(len(ep_experience)):
            if hind_goal is None:   # future state strategy
                future = np.random.randint(i, len(ep_experience))
                goal_ = ep_experience[future][3]
            else:
                goal_ = hind_goal

            state_ = ep_experience[i][0]
            action_ = ep_experience[i][1]
            next_state_ = ep_experience[i][3]

            if extract_feature:     
                done_, reward_ = self.her_reward_func_1(next_state_, goal_)
            else:
                done_, reward_ = self.her_reward_func_2(next_state_, goal_)

            # add new experience to the main buffer
            self.buffer.record([state_, action_, reward_, next_state_, done_, goal_])

    def save_model(self, save_path):
        os.makedirs(save_path, exist_ok=True)
        actor_file = save_path + 'sac_actor_wts.h5'
        critic1_file = save_path + 'sac_c1_wts.h5'
        critic2_file = save_path + 'sac_c2_wts.h5'
        target_c1_file = save_path + 'sac_c1t_wts.h5'
        target_c2_file = save_path + 'sac_c2t_wts.h5'
        self.actor.save_weights(actor_file)
        self.critic1.save_weights(critic1_file)
        self.critic2.save_weights(critic2_file)
        self.target_critic1.save_weights(target_c1_file)
        self.target_critic2.save_weights(target_c2_file)

    def load_model(self, load_path):
        actor_file = load_path + 'sac_actor_wts.h5'
        critic1_file = load_path + 'sac_c1_wts.h5'
        critic2_file = load_path + 'sac_c2_wts.h5'
        target_c1_file = load_path + 'sac_c1t_wts.h5'
        target_c2_file = load_path + 'sac_c2t_wts.h5'
        self.actor.load_weights(actor_file)
        self.critic1.load_weights(critic1_file)
        self.critic2.load_weights(critic2_file)
        self.target_critic1.load_weights(target_c1_file)
        self.target_critic2.load_weights(target_c2_file)

