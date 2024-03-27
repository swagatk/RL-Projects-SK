'''
Solving Atari Games
# install the following packages
!pip install gymnasium[atari] >/dev/null 2>&1
!pip install gymnasium[accept-rom-license] >/dev/null 2>&1

'''

import gymnasium as gym
import numpy as np
import sys
import cv2
import imageio
from PIL import Image
import PIL.ImageDraw as ImageDraw
import matplotlib.pyplot as plt 
import tensorflow as tf 
import keras
import datetime
import os

sys.path.append('/home/kumars/RL-Projects-SK')
from src.algo.dqn import DQNAgent, DQNPERAgent


def _label_with_episode_number(frame, episode_num, step_num):
    im = Image.fromarray(frame)
    drawer = ImageDraw.Draw(im)
    if np.mean(im) < 128: # for dark image
        text_color = (255, 255, 255)
    else:
        text_color = (0, 0, 0)
    drawer.text((im.size[0]/20, im.size[1]/18),
                f'Episode: {episode_num+1}, Steps: {step_num+1}',
                fill=text_color)
    return im

class DQNAtariAgent(DQNPERAgent):
    def __init__(self, obs_shape: tuple, n_actions: int,
                        buffer_size=2000, batch_size=24,
                        ddqn_flag=True, model=None, per_flag=True):
        self.per_flag = per_flag
        
        if self.per_flag:
            super().__init__(obs_shape, n_actions, buffer_size,
                        batch_size, ddqn_flag, model)
        else:
            DQNAgent.__init__(self, obs_shape, n_actions, buffer_size, 
                              batch_size, ddqn_flag, model)
        
    def experience_replay(self):
        if self.per_flag:
            super().experience_replay()
        else:
            DQNAgent.experience_replay(self)

    def preprocess(self, observation, x_crop=(1, 172), y_crop=None):

        output_shape = self.obs_shape[:-1] # all but last

        # crop image
        if x_crop is not None and y_crop is not None:
            xlow, xhigh = x_crop
            ylow, yhigh = y_crop
            observation = observation[xlow:xhigh, ylow:yhigh]
        elif x_crop is not None and y_crop is None:
            xlow, xhigh = x_crop
            observation = observation[xlow:xhigh, :]
        elif x_crop is None and y_crop is not None:
                ylow, yhigh = y_crop
                observation = observation[:, ylow:yhigh]
        else:
            observation = observation

        # resize image
        if output_shape is not None:
            observation = cv2.resize(observation, output_shape)

        # normalize image
        observation = observation / 255.  # normalize between 0 & 1
        if observation.ndim == 2:
            return np.reshape(observation, output_shape + (1,))
        elif observation.ndim == 3:
            return observation
        else:
            raise ValueError('Supports only 2 or 3D images')
            
    def train(self, env, max_episodes=300, 
          train_freq=1, copy_freq=1, 
          filename=None, 
          tb_fw=None):
    
        if filename is not None:
            file = open(filename, 'w')

        # choose between soft and hard update
        tau = 0.01 if copy_freq < 10 else 1.0
        best_score, global_step_cnt = 0, 0
        scores, avg_scores, avg100_scores = [], [], []
        global_step_cnt = 0
        for e in range(max_episodes):
            state = env.reset()[0]
            state = self.preprocess(state)
            state = np.expand_dims(state, axis=0)
            done = False
            ep_reward = 0
            while not done:
                global_step_cnt += 1
                # take action
                action = self.get_action(state)
                # collect reward
                next_state, reward, done, _, _ = env.step(action)
                next_state = self.preprocess(next_state)
                next_state = np.expand_dims(next_state, axis=0) # (-1, 4)
                # store experiences in eplay buffer
                self.store_experience(state, action, reward, next_state, done)
                state = next_state
                ep_reward += reward
                # train
                if global_step_cnt % train_freq == 0:
                    self.experience_replay()

                # update target model
                if global_step_cnt % copy_freq == 0:
                    self.update_target_model(tau=tau)
                # end of while-loop
            if ep_reward > best_score:
                self.save_model('best_model.h5')
                best_score = ep_reward
            scores.append(ep_reward)
            avg_scores.append(np.mean(scores))
            avg100_scores.append(np.mean(scores[-100:]))
            if filename is not None:
                file.write(f'{e}\t{ep_reward}\t{np.mean(scores)}\t{np.mean(scores[-100:])}\n')
            if tb_fw is not None:
                tf.summary.scalar('ep_reward', data=ep_reward, step=e)
                tf.summary.scalar('avg_ep_reward', data=np.mean(scores), step=e)
                tf.summary.scalar('epsilon', data=self.get_epsilon(), step=e)
            print(f'\re:{e}, ep_reward: {ep_reward}, avg_ep_reward: {np.mean(scores):.2f}', end="")
            sys.stdout.flush()
        # end of for loop
        print('End of training')
        if filename is not None:
            file.close()
            
    def validate(self, env, wt_file=None, save_gif=True):
        if wt_file is not None:
            self.load_model(wt_file)
        frames, scores = [], []
        for i in range(10):
            print('\r episode: ', i, end="")
            sys.stdout.flush()
            state = env.reset()[0]
            state = self.preprocess(state)
            state = np.expand_dims(state, axis=0)
            step = 0
            ep_reward = 0
            while True:
                step += 1
                if save_gif and env.render_mode == 'rgb_array':
                    frame = env.render()
                    frames.append(_label_with_episode_number(frame, i, step))
                action = self.get_action(state)
                next_state, reward, done, _, _ = env.step(action)
                next_state = self.preprocess(next_state)
                next_state = np.expand_dims(next_state, axis=0)
                state = next_state
                ep_reward += reward
                if done:
                    scores.append(ep_reward)
                    if save_gif and env.render_mode == 'rgb_array':
                        frame = env.render()
                        frames.append(_label_with_episode_number(frame, i, step))
                break
            # while-loop ends here
        # for-loop ends here
        if save_gif:
            imageio.mimwrite(os.path.join('./', 'dqn_pacman.gif'), frames, duration=1000/60)
        print('\nAverage episodic score: ', np.mean(scores))


if __name__ == '__main__':

    # set up tensorboard log dir
    logdir = os.path.join("logs", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    fw = tf.summary.create_file_writer(logdir + "/metrics")
    fw.set_as_default()

    # create gym environment
    env = gym.make('ALE/MsPacman-v5', obs_type="grayscale")
    obs_shape = (84, 84, 1) 
    n_actions = env.action_space. n

    print('shape of action space: ', env.action_space.n)
    print('shape of observation space: ', env.observation_space.shape)
    

    # create a DQN model
    model = keras.Sequential([
        keras.layers.Conv2D(32, kernel_size=8, strides=4, padding='same',
                            activation='relu', kernel_initializer='he_uniform',
                            input_shape=obs_shape),
        keras.layers.MaxPooling2D(pool_size=(2,2)),
        keras.layers.Conv2D(64, kernel_size=2, strides=1, padding='same',
                            activation='relu', kernel_initializer='he_uniform'),
        keras.layers.MaxPooling2D(pool_size=(2,2)),
        keras.layers.Flatten(),
        keras.layers.Dense(128, activation='relu', kernel_initializer='he_uniform'),
        keras.layers.Dense(n_actions, activation='linear')
    ])
    model.compile(loss='mse', optimizer="adam")

    # create an agent
    agent = DQNAtariAgent(obs_shape, n_actions, 
                      buffer_size=60000, batch_size=64, 
                      model=model, per_flag=False)
    
    # train the agent
    agent.train(env, max_episodes=10000, copy_freq=100, 
                filename='pacman_dqn.txt', tb_fw=fw)
    fw.close()



 

