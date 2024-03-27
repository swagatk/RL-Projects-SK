import sys
import tensorflow as tf 
import os
import imageio
import numpy as np
from PIL import Image
import PIL.ImageDraw as ImageDraw
import matplotlib.pyplot as plt
import gymnasium as gym 
import keras 
import gymnasium as gym
import random

sys.path.append('/home/kumars/RL-Projects-SK/src')
from algo.dqn import DQNAgent, DQNPERAgent
##################

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

############################
# training function
def train(env, agent, max_episodes=300,
          train_freq=1, copy_freq=10):
    file = open('mc_dqn.txt', 'w')
    tau = 0.1 if copy_freq < 10 else 1.0
    max_steps = 200
    car_positions = []
    scores, avg_scores = [], []
    global_step_cnt = 0
    for e in range(max_episodes):
        state = env.reset()[0]
        state = np.expand_dims(state, axis=0)
        done = False
        ep_reward = 0
        t = 0
        max_pos = -99.0
        while not done:
            global_step_cnt += 1
            # take action
            action = agent.get_action(state)
            # receive rewards
            next_state, reward, done, _, _ = env.step(action)
            next_state = np.expand_dims(next_state, axis=0)
            # engineer rewards for better learning
            if next_state[0][0] >=0.5:
                reward += 200
            else:
                reward = 5 * abs(next_state[0][0] - state[0][0]) + 3 * abs(state[0][1])
            # track maximum position achieved
            if next_state[0][0] > max_pos:
                max_pos = next_state[0][0]

            # store experiences
            agent.store_experience(state, action, reward, next_state, done)
            state = next_state
            ep_reward += reward
            t += 1

            # train
            if global_step_cnt % train_freq == 0:
                #agent.experience_replay()
                agent.experience_replay()

            # update target model
            if global_step_cnt % copy_freq == 0:
                agent.update_target_model(tau=tau)

            if done and t < max_steps: # success
                print('\n Successfully solved the problem in {} epsisodes, \
                                max_pos: {:.2f}, steps: {}\n'.format(e, max_pos, t))
                agent.save_model('best_model_{}.weights.h5'.format(e))

            if t >= max_steps: # failure
                break
        # episode ends here
        car_positions.append(state[0][0])
        scores.append(ep_reward)
        avg_scores.append(np.mean(scores))
        epsilon = agent.get_epsilon()
        file.write(f'{e}\t{ep_reward:.2f}\t{np.mean(scores):.2f}\t{max_pos:.2f}\t{t}\t{epsilon:.2f}\n' )
        print(f'\re:{e}, ep_reward: {ep_reward:.2f}, avg ep reward: {np.mean(scores):.2f}, ep_steps: {t}, max_pos: {max_pos:.2f}, epsilon: {epsilon:.2f}', end="")
        sys.stdout.flush()
    print('End of training')
    file.close()

###########################
# generates gif animation file    
def validate(env, agent, wt_file: None):
    if wt_file is not None:
        agent.load_model(wt_file)

    max_steps = 200
    frames = []
    scores = []
    for i in range(10):
        #ipdb.set_trace()
        print('\r episode: ', i, end="")
        sys.stdout.flush()
        state = env.reset()[0]
        state = np.expand_dims(state, axis=0)
        step = 0
        ep_reward = 0
        while step < max_steps: 
            step += 1
            frame = env.render()
            frames.append(_label_with_episode_number(frame, i, step))
            action = agent.get_action(state, epsilon=0.0001)
            next_state, reward, done, _, _ = env.step(action)
            next_state = np.expand_dims(next_state, axis=0)
            state = next_state
            ep_reward += reward
            if done:
                frame = env.render()
                frames.append(_label_with_episode_number(frame, i, step))
                break
        # end of episode
        scores.append(ep_reward)
    # for-loop ends here
    imageio.mimwrite(os.path.join('./', 'dqn_mc.gif'), frames, duration=1000/60)
    print('\nAverage episodic score: ', np.mean(scores))

def generate_plot(filename):
    import pandas as pd
    df = pd.read_csv(filename, sep='\t')
    df.columns=['episode', 'ep_reward', 'avg_reward', 'car_position', 'ep_steps', 'epsilon'] 
    fig, axes = plt.subplots(3)
    df.plot(x='episode', y='ep_reward', ax=axes[0])
    df.plot(x='episode', y='avg_reward', ax=axes[0])
    axes[0].legend(loc='best')
    axes[0].grid()
    axes[0].set_ylabel('rewards')
    # plot car positions separately
    df.plot(x='episode', y='car_position', ax=axes[1])
    axes[1].set_ylabel('car positions')
    axes[1].grid()
    df.plot(x='episode', y='ep_steps', ax=axes[2])
    axes[2].set_ylabel('Steps per episode')
    axes[2].grid()
    plt.savefig('mc_dqn.png')


#########################
if __name__ == '__main__':

    # for reproducibility
    # keras.utils.set_random_seed(42)  # sets seeds for base-python, numpy and tf
    # tf.config.experimental.enable_op_determinism()
    # tf.random.set_seed(42)
    # np.random.seed(42)
    # random.seed(42)

    # create a gym environment
    env = gym.make('MountainCar-v0', render_mode='rgb_array')
    obs_shape = env.observation_space.shape
    action_shape = env.action_space.shape
    n_actions = env.action_space.n
    print('Observation shape: ', obs_shape)
    print('Action shape: ', action_shape)
    print('Action size: ', n_actions)
    print('Max episodic steps: ', env.spec.max_episode_steps)

    # create a model
    model = keras.Sequential([
        keras.layers.Dense(30, input_shape=obs_shape, activation='relu',
                        kernel_initializer='he_uniform'),
        keras.layers.Dense(60, activation='relu', kernel_initializer='he_uniform'),
        keras.layers.Dense(n_actions, activation='linear',
                        kernel_initializer='he_uniform')
    ])
    model.compile(loss='mse', optimizer=keras.optimizers.Adam())

    # create a DQN agent
    agent = DQNPERAgent(obs_shape, n_actions,
                    buffer_size=20000,
                    batch_size=64,
                    model=model)

    # agent = DQNAgent(obs_shape, n_actions,
    #                 buffer_size=20000,
    #                 batch_size=64,
    #                 model=model)

    # train the model
    train(env, agent, max_episodes=100, copy_freq=100)
    
    # use the best model to create animation
    #validate(env, agent, wt_file='best_model_99.weights.h5')

    # generate plot
    #generate_plot('mc_dqn.txt')




