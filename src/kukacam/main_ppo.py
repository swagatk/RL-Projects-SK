"""
PPO algorithm for Kuka Diverse Object Environment
Status: Not working. I did not wait till 150K iterations. I did not see any change during first 16K episodes.
"""
from pybullet_envs.bullet.kuka_diverse_object_gym_env import KukaDiverseObjectEnv
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from ppo import KukaPPOAgent
import datetime
import pickle
import tensorboard

if __name__ == '__main__':

    #####################
    # TENSORBOARD SETTINGS
    print('Tensorboard version:', tensorboard.__version__)
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    train_log_dir = 'logs/train/' + current_time
    graph_log_dir = 'logs/func/' + current_time
    train_summary_writer = tf.summary.create_file_writer(train_log_dir)

    with tf.Graph().as_default():
        print(tf.executing_eagerly())
    ######################

    # start open/AI GYM environment
    #env = KukaCamGymEnv(renders=False, isDiscrete=False)
    env = KukaDiverseObjectEnv(renders=False,
                               isDiscrete=False,
                               maxSteps=20,
                               removeHeightHack=False)
    print('shape of Observation space: ', env.observation_space.shape)
    print('shape of Action space: ', env.action_space.shape)
    print('Reward Range: ', env.reward_range)
    print('Action High value: ', env.action_space.high)
    print('Action Low Value: ', env.action_space.low)

    ################
    # Hyper-parameters
    ######################
    MAX_EPISODES = 1500001  #15000 * 100

    LR_A = 0.001
    LR_C = 0.002
    GAMMA = 0.99
    LAMBDA = 0.95  # required for GAE
    EPSILON = 0.2  # required for PPO-CLIP
    EPOCHS = 100    # check ??

    MEMORY_CAPACITY = 2000  # max permissible on my machine is 50000
    BATCH_SIZE = 125
    UPDATE_FREQ = 500
    TRG_EPOCHS = 10   # training epochs

    # Directory to store intermediate models
    SAVE_DIR = './chkpt/'

    # load saved models from this directory
    LOAD_DIR = './reload/'
    LOAD_MODEL = False

    SAVE_FREQ = 2000

    action_upper_bound = env.action_space.high
    state_size = env.observation_space.shape  # (48, 48, 3)
    action_size = env.action_space.shape  # (3,)

    print('state_size: ', state_size)
    print('action_size: ', action_size)

    # Create a Kuka Actor-Critic Agent
    agent = KukaPPOAgent(state_size, action_size,
                       LR_A, LR_C,
                       BATCH_SIZE,
                       MEMORY_CAPACITY, LAMBDA,
                       EPSILON, GAMMA, action_upper_bound,
                       UPDATE_FREQ, TRG_EPOCHS)

    # if loading from previous models

    if LOAD_MODEL:
        agent.load_model(LOAD_DIR, 'actor_weights.h5', 'critic_weights.h5',
                         'replay_buffer.dat')

        load_file = LOAD_DIR + 'ep_reward.dat'
        with open(load_file, 'rb') as file:
            load_param = pickle.load(file)

        start_episode = load_param[0]
        ep_reward_list = load_param[1]
        avg_reward_list = load_param[2]

    else:
        start_episode = 0
        ep_reward_list = []
        avg_reward_list = []

    actor_loss, critic_loss = 0, 0
    best_score = - np.inf
    t = 0           # total number of training steps
    print('Main training loop')
    for episode in range(start_episode, MAX_EPISODES):
        obsv = env.reset()
        state = np.asarray(obsv, dtype=np.float32) / 255.0  # convert into float array
        ep_reward = 0
        ep_adv = 0  # advantage for each episode
        frames = []
        steps = 0
        while True:
            if episode > MAX_EPISODES - 3:
                frames.append(env.render(mode='rgb_array'))

            # take an action as per the policy
            action = agent.policy(state)

            # obtain next state and rewards
            next_obsv, reward, done, info = env.step(action)
            next_state = np.asarray(next_obsv, dtype=np.float32) / 255.0  # convert into float array
            tb_img = np.reshape(next_state, (-1,) + state_size)  # for tensorboard
            ep_reward += reward

            with train_summary_writer.as_default():
                tf.summary.image("Training Image", tb_img, step=episode)
                tf.summary.histogram("action_vector", action, step=steps)

            # store experiences in a memory buffer
            experience = (state, action, reward, next_state, done)
            agent.record(experience)

            # train the network
            actor_loss, critic_loss = agent.experience_replay()

            with train_summary_writer.as_default():
                tf.summary.scalar('actor_loss', actor_loss, step=episode)
                tf.summary.scalar('critic_loss', critic_loss, step=episode)

            state = next_state
            steps += 1
            t += 1

            if done:
                if episode > 500 and ep_reward > best_score:
                    best_score = ep_reward
                    agent.save_model(SAVE_DIR, 'best_actor.h5',
                                     'best_critic.h5', 'best_replay.dat')
                break

        ep_reward_list.append(ep_reward)
        avg_reward = np.mean(ep_reward_list[-100:])
        avg_reward_list.append(avg_reward)
        print("Episode: {}, Buffer size: {}, reward = {}, Avg Reward = {} ".format(episode,
                                            len(agent.buffer), ep_reward, avg_reward))

        with open(SAVE_DIR + 'episode_reward.txt', 'a+') as file:
            file.write('{}\t{}\t{}\n'.format(episode, ep_reward, avg_reward))

        with train_summary_writer.as_default():
            tf.summary.scalar('avg_reward', avg_reward, step=episode)

        if episode % SAVE_FREQ == 0:
            agent.save_model(SAVE_DIR, 'actor_weights.h5',
                             'critic_weights.h5', 'replay_buffer.dat')

            save_param = [episode, ep_reward_list, avg_reward_list]
            with open(SAVE_DIR+'ep_reward.dat', 'wb') as file:
                pickle.dump(save_param, file)

    env.close()


