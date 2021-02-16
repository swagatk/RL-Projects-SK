'''
Main File Kuka Environment
'''
#from pybullet_envs.bullet.kukaCamGymEnv import KukaCamGymEnv
from pybullet_envs.bullet.kuka_diverse_object_gym_env import KukaDiverseObjectEnv
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
# from kuka_actor_critic import KukaActorCriticAgent
from ddpg_PER import DDPG_PER_Agent
#from kuka_actor_critic4 import KukaTD3Agent
#from ddpg import DDPG_Agent
import datetime
import pickle

if __name__ == '__main__':

    #####################
    # TENSORBOARD SETTINGS
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
    MAX_EPISODES = 15001

    LR_A = 0.001
    LR_C = 0.002
    GAMMA = 0.99

    replacement = [
        dict(name='soft', tau=0.005),
        dict(name='hard', rep_iter_a=600, rep_iter_c=500)
    ][0]  # you can try different target replacement strategies

    MEMORY_CAPACITY = 50000  # max permissible on my machine is 50000
    BATCH_SIZE = 128

    # episodes for random exploration
    RAND_EPS = 0

    # Directory to store intermediate models
    SAVE_DIR = './chkpt/'

    # load saved models from this directory
    LOAD_DIR = './reload/'
    LOAD_MODEL = False

    SAVE_FREQ = 500

    # priority experience replay
    __PER = True

    upper_bound = env.action_space.high
    lower_bound = env.action_space.low
    state_size = env.observation_space.shape  # (48, 48, 3)
    action_size = env.action_space.shape  # (3,)

    print('state_size: ', state_size)
    print('action_size: ', action_size)

    # Create a Kuka Actor-Critic Agent
    agent = DDPG_PER_Agent(state_size, action_size,
                        replacement, LR_A, LR_C,
                        BATCH_SIZE,
                        MEMORY_CAPACITY,
                        GAMMA,
                        upper_bound, lower_bound)

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
    print('Main training loop')
    for episode in range(start_episode, MAX_EPISODES):
        obsv = env.reset()
        state = np.asarray(obsv, dtype=np.float32) / 255.0  # convert into float array
        episodic_reward = 0
        frames = []
        steps = 0
        while True:
            if episode > MAX_EPISODES - 3:
                frames.append(env.render(mode='rgb_array'))

            # take an action as per the policy
            if episode < RAND_EPS:  # explore for some episodes
                action = env.action_space.sample()
            else:
                action = agent.policy(state)

            # obtain next state and rewards

            next_obsv, reward, done, info = env.step(action)
            next_state = np.asarray(next_obsv, dtype=np.float32) / 255.0  # convert into float array

            #tb_img = np.reshape(next_state, (-1, 48, 48, 3))  # for tensorboard
            tb_img = np.reshape(next_state, (-1,) + state_size)  # for tensorboard

            with train_summary_writer.as_default():
                tf.summary.image("Training Image", tb_img, step=episode)
                tf.summary.histogram("action_vector", action, step=steps)

            episodic_reward += reward

            # print('reward:', episodic_reward)
            experience = (state, action, reward, next_state, done)

            # store experience
            agent.record(experience)

            # train the network
            actor_loss, critic_loss = agent.experience_replay()

            # update the target model
            agent.update_targets()

            with train_summary_writer.as_default():
                tf.summary.scalar('actor_loss', actor_loss, step=episode)
                tf.summary.scalar('critic_loss', critic_loss, step=episode)

            state = next_state
            steps += 1

            if done:
                if episode > 500 and episodic_reward > best_score:
                    best_score = episodic_reward
                    agent.save_model(SAVE_DIR, 'best_actor.h5',
                                     'best_critic.h5', 'best_replay.dat')
                break

        ep_reward_list.append(episodic_reward)
        avg_reward = np.mean(ep_reward_list[-100:])
        avg_reward_list.append(avg_reward)
        print("Episode: {}, Buffer size: {}, reward = {}, Avg Reward = {} ".format(episode,
                                            len(agent.buffer), episodic_reward, avg_reward))

        with open(SAVE_DIR + 'episode_reward.txt', 'a+') as file:
            file.write('{}\t{}\t{}\n'.format(episode, episodic_reward, avg_reward))

        with train_summary_writer.as_default():
            tf.summary.scalar('avg_reward', avg_reward, step=episode)

        if episode % SAVE_FREQ == 0:
            agent.save_model(SAVE_DIR, 'actor_weights.h5',
                             'critic_weights.h5', 'replay_buffer.dat')

            save_param = [episode, ep_reward_list, avg_reward_list]
            with open(SAVE_DIR+'ep_reward.dat', 'wb') as file:
                pickle.dump(save_param, file)

            if __PER:
                agent.buffer.save_priorities_txt(SAVE_DIR + 'priorities.txt')

    env.close()

    # plot
    plt.plot(avg_reward_list)
    plt.xlabel('Episodes')
    plt.ylabel('Avg episodic reward')
    plt.grid()
    plt.savefig('./kuka_ddpg_tf2.png')
    plt.show()
    # save animation as GIF
    # save_frames_as_gif(frames)

