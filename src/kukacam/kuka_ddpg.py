'''
Applying DDPG algorithm to KukaCamGym Environment

'''
import gym
#from pybullet_envs.bullet.kukaCamGymEnv import KukaCamGymEnv
from pybullet_envs.bullet.kuka_diverse_object_gym_env import KukaDiverseObjectEnv
import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from kuka_actor_critic import KukaACAgent
from utils import *
import datetime
print("Tensorflow Version: ", tf.__version__)

if __name__ == '__main__':


    #####################
    # TENSORBOARD SETTINGS
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    train_log_dir = 'logs/train/' + current_time
    train_summary_writer = tf.summary.create_file_writer(train_log_dir)
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
    MAX_EPISODES = 50000
    STACK_SIZE = 5 # number of frames stacked together - not used

    LR_A = 0.001
    LR_C = 0.002
    GAMMA = 0.99

    replacement = [
        dict(name='soft', tau=0.005),
        dict(name='hard', rep_iter_a=600, rep_iter_c=500)
    ][0]  # you can try different target replacement strategies

    MEMORY_CAPACITY = 70000
    BATCH_SIZE = 128

    upper_bound = env.action_space.high
    lower_bound = env.action_space.low
    state_size = env.observation_space.shape
    action_size = env.action_space.shape[0]

    print('state_size: ', state_size)
    print('action_size: ', action_size)

    # Create a Kuka Actor-Critic Agent
    agent = KukaACAgent(state_size, action_size,
                     replacement, LR_A, LR_C,
                     BATCH_SIZE,
                     MEMORY_CAPACITY,
                     GAMMA,
                     upper_bound, lower_bound)

    actor_loss, critic_loss = 0, 0
    ep_reward_list = []
    avg_reward_list = []
    best_score = - np.inf
    for episode in range(MAX_EPISODES):
        print('Episode = ', episode)
        obsv = env.reset()
        state = np.asarray(obsv, dtype=np.float32) / 255.0
        #plt.imshow(state)
        #plt.show()
        episodic_reward = 0
        frames = []


        step = 0
        while True:
            if episode > MAX_EPISODES - 3:
                frames.append(env.render(mode='rgb_array'))

            # convert the numpy array state into a tensor of size (1, 48, 48)
            tf_state = tf.expand_dims(tf.convert_to_tensor(state), 0)

            # take an action as per the policy
            action = agent.policy(tf_state)

            # obtain next state and rewards
            next_state, reward, done, info = env.step(action)

            #tb_img = np.reshape(next_state, (-1, 48, 48, 3))  # for tensorboard
            tb_img = np.reshape(next_state, (-1,) + state_size) # for tensorboard

            with train_summary_writer.as_default():
                tf.summary.image("Training Image", tb_img, step=episode)
            episodic_reward += reward

            # print('reward:', episodic_reward)

            # store experience
            agent.buffer.record(state, action, reward, next_state)

            # train the network
            actor_loss, critic_loss = agent.experience_replay()
            with train_summary_writer.as_default():
                tf.summary.scalar('actor_loss', actor_loss, step=episode)
                tf.summary.scalar('critic_loss', critic_loss, step=episode)

            # update the target model
            agent.update_targets()

            state = next_state
            step += 1
            #print('step: ', step)

            if done:
                if episodic_reward > best_score:
                    best_score = episodic_reward
                    agent.actor.save_weights('./kuka_actor_weights.h5')
                    agent.critic.save_weights('./kuka_actor_weights.h5')
                break

        ep_reward_list.append(episodic_reward)
        avg_reward = np.mean(ep_reward_list[-100:])
        print("Episode * {} * Avg Reward = {} ".format(episode, avg_reward))
        avg_reward_list.append(avg_reward)

        with train_summary_writer.as_default():
            tf.summary.scalar('avg_reward', avg_reward, step=episode)

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

