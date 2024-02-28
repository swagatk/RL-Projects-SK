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
from dqn import DQNAgent
import datetime

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
def train(env, agent, max_episodes=300,
          train_freq=1, copy_freq=1,
          file_writer=None):

    file = open('mc_dqn.txt', 'w')
    file.write('episode\tepisodic_reward\tcar_position\tavg_score\n')
    if copy_freq < 10:
        tau = 0.1
    else:
        tau = 1.0

    best_score = 0
    car_positions = []
    scores = []
    avg_scores, avg100_scores = [], []
    global_step_cnt = 0
    for e in range(max_episodes):
        state = env.reset()[0]
        state = np.expand_dims(state, axis=0)
        done = False
        ep_reward = 0
        t = 0
        while not done:
            global_step_cnt += 1
            action = agent.get_action(state)
            next_state, reward, done, _, _ = env.step(action)
            # if terminal_reward_penalty[0]:
            #   reward = reward if not done else terminal_reward_penalty[1]

            if next_state[0] >=0.5:
                reward += 20   

            next_state = np.expand_dims(next_state, axis=0) # (-1, 4)
            agent.store_experience(state, action, reward, next_state, done)
            state = next_state
            ep_reward += reward
            t += 1
            # print(f'\r{t}', end="")
            # sys.stdout.flush()

            # train
            if global_step_cnt % train_freq == 0:
                agent.experience_replay_4()

            # update target model
            if global_step_cnt % copy_freq == 0:
                agent.update_target_model(tau=tau)

            if t >= 200:
                break
            # episode ends here

        if ep_reward > best_score:
            best_score = ep_reward
            agent.save_model('best_model.h5')
        car_positions.append(state[0][0])
        scores.append(ep_reward)
        avg_scores.append(np.mean(scores))
        avg100_scores.append(np.mean(scores[-100:]))
        file.write(f'{e}\t{ep_reward}\t{state[0][0]}\t{np.mean(scores)}\n' )
        if file_writer is not None:
            tf.summary.scalar('episodic_reward', data=ep_reward, step=e)
            tf.summary.scalar('avg_ep_reward', data=np.mean(scores), step=e)
            tf.summary.scalar('car_position',data=np.mean(state[0][0]), step=e)
            file_writer.flush()
        print(f'\re:{e}, episodic reward: {ep_reward}, avg ep reward: {np.mean(scores)}', end="")
        sys.stdout.flush()
    print('End of training')
    file.close()


#############
def validate(env, agent, wt_file: None):
    if wt_file is not None:
        agent.load_model(wt_file)
    frames = []
    scores = []
    for i in range(10):
        print('\r episode: ', i, end="")
        sys.stdout.flush()
        state = env.reset()[0]
        state = np.expand_dims(state, axis=0)
        step = 0
        while True:
            step += 1
            frame = env.render()
            frames.append(_label_with_episode_number(frame, i, step))
            action = agent.get_action(state)
            next_state, reward, done, _, _ = env.step(action)
            next_state = np.expand_dims(next_state, axis=0)
            state = next_state
            if done:
                scores.append(step)
                frame = env.render()
                frames.append(_label_with_episode_number(frame, i, step))
                break
    # for-loop ends here
    imageio.mimwrite(os.path.join('./', 'dqn_mc.gif'), frames, duration=1000/60)
    print('\nAverage episodic score: ', np.mean(scores))

########
  
if __name__ == '__main__':
  
    env = gym.make('MountainCar-v0')
    obs_shape = env.observation_space.shape
    action_shape = env.action_space.shape
    n_actions = env.action_space.n
    print('Observation shape: ', obs_shape)
    print('Action shape: ', action_shape)
    print('Action size: ', n_actions)
    print('Max episodic steps: ', env.spec.max_episode_steps)

    # tensorboard logging
    logdir = os.path.join("logs", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    file_writer = tf.summary.create_file_writer(logdir + "/metrics")
    file_writer.set_as_default()

    # create a model
    model = keras.Sequential([
        keras.layers.Dense(30, input_shape=obs_shape, activation='relu'),
        keras.layers.Dense(60, activation='relu'),
        keras.layers.Dense(n_actions, activation='linear')
    ])
    model.compile(loss='mse', optimizer='adam')
    
    # create DQN Agent
    agent = DQNAgent(obs_shape, n_actions,
                 buffer_size=20000, 
                 batch_size=64,
                 model=model)
    
    # train
    train(env, agent, max_episodes=1000, copy_freq=1, 
          file_writer=file_writer)