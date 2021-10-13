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
    def __init__(self, **kwargs):
        super().__init__(**kwargs)


    def validate(self, max_eps=50):
        ep_reward_list = []
        for ep in range(max_eps):
            obs = self.env.reset()

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

    def run(self, train_freq=5):

        if self.filename is not None: 
            self.filename = uniquify(self.path + self.filename)

        # initial state
        obs = self.env.reset()
        if self.image_input:
            state = np.asarray(obs['observation'], dtype=np.float32) / 255.0
            goal = np.asarray(obs['desired_goal_img'], dtype=np.float32) / 255.0
        else:
            state = obs['observation']
            goal = obs['desired_goal']

        start = datetime.datetime.now()
        val_scores = []                 # validation scores 
        best_score = -np.inf
        ep_lens = []        # episodic length
        ep_scores = []      # All episodic scores
        s_scores = []       # season scores
        self.episodes = 0       # total episode count
        ep_actor_losses = []    # actor losses
        ep_critic_losses = []   # critic losses

        for s in range(self.seasons):
            ep_experience = []    # required for HER
            s_score = 0         # season score 
            ep_len = 0          # episode length
            ep_score = 0        # score for each episode
            ep_cnt = 0          # no. of episodes in each season
            done = False
            for t in range(self.training_batch):
                action, _ = self.policy(state, goal)
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
                
                if done:
                    s_score += ep_score
                    ep_cnt += 1     # no. of episodes in each season
                    self.episodes += 1  # total episode count
                    ep_scores.append(ep_score)
                    ep_lens.append(ep_len) 

                    # HER strategies
                    hind_goal = achieved_goal

                    self.add_her_experience(ep_experience, hind_goal, 
                                                self.use_her['extract_feature'])
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
                    obs = self.env.reset()
                    if self.image_input:
                        state = np.asarray(obs['observation'], dtype=np.float32) / 255.0
                        goal = np.asarray(obs['desired_goal_img'], dtype=np.float32) / 255.0
                    else:
                        state = obs['observation']
                        goal = obs['desired_goal']

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
