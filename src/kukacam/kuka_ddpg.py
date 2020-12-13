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
# from kuka_actor_critic import KukaActorCriticAgent
#from kuka_actor_critic2 import KukaActorCriticAgent
from kuka_actor_critic3 import KukaActorCriticAgent
import datetime


if __name__ == '__main__':


    #####################
    # TENSORBOARD SETTINGS
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    train_log_dir = 'logs/train/' + current_time
    graph_log_dir = 'logs/func/' + current_time
    train_summary_writer = tf.summary.create_file_writer(train_log_dir)
    #graph_summary_writer = tf.summary.create_file_writer(graph_log_dir)
    #tf.summary.trace_on(graph=True, profiler=True)

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
    MAX_EPISODES = 10000

    LR_A = 2e-4
    LR_C = 2e-4
    GAMMA = 0.99

    replacement = [
        dict(name='soft', tau=0.005),
        dict(name='hard', rep_iter_a=600, rep_iter_c=500)
    ][0]  # you can try different target replacement strategies

    MEMORY_CAPACITY = 20000
    BATCH_SIZE = 128

    __PER = True  # priority experience replay

    upper_bound = env.action_space.high
    lower_bound = env.action_space.low
    state_size = env.observation_space.shape  # (48, 48, 3)
    action_size = env.action_space.shape  # (3,)

    print('state_size: ', state_size)
    print('action_size: ', action_size)

    # Create a Kuka Actor-Critic Agent
    agent = KukaActorCriticAgent(state_size, action_size,
                        replacement, LR_A, LR_C,
                        BATCH_SIZE,
                        MEMORY_CAPACITY,
                        GAMMA,
                        upper_bound, lower_bound)

    # Pre-train: Fill the replay-buffer with random experiences
    print('Pretrain Phase ... Wait')
    for ep in range(200):
        obsv = env.reset()
        state = np.asarray(obsv, dtype=np.float32) / 255.0  # convert into float array
        action = env.action_space.sample()
        next_obsv, reward, done, info = env.step(action)
        next_state = np.asarray(next_obsv, dtype=np.float32) / 255.0
        experience = (state, action, reward, next_state)
        if __PER:
            agent.buffer.record(experience, 0.1)
        else:
            agent.buffer.record(experience)
    print('Pre-train completed')

    actor_loss, critic_loss = 0, 0
    ep_reward_list = []
    avg_reward_list = []
    best_score = - np.inf
    print('Main training loop')
    for episode in range(MAX_EPISODES):
        print('Episode = ', episode)
        obsv = env.reset()
        state = np.asarray(obsv, dtype=np.float32) / 255.0  # convert into float array
        episodic_reward = 0
        frames = []
        step = 0
        while True:
            if episode > MAX_EPISODES - 3:
                frames.append(env.render(mode='rgb_array'))

            # take an action as per the policy
            action = agent.policy(state)

            # obtain next state and rewards
            next_obsv, reward, done, info = env.step(action)
            next_state = np.asarray(next_obsv, dtype=np.float32) / 255.0  # convert into float array

            #tb_img = np.reshape(next_state, (-1, 48, 48, 3))  # for tensorboard
            tb_img = np.reshape(next_state, (-1,) + state_size) # for tensorboard

            with train_summary_writer.as_default():
                tf.summary.image("Training Image", tb_img, step=episode)
                tf.summary.histogram("action_vector", action, step=step)

            episodic_reward += reward

            # print('reward:', episodic_reward)
            experience = (state, action, reward, next_state)
            priority = agent.get_per_error(experience)

            # store experience
            if __PER:
                agent.buffer.record(experience, priority)
            else:
                agent.buffer.record(experience)

            # train the network
            actor_loss, critic_loss = agent.experience_replay()

            # update the target model
            agent.update_targets()

            with train_summary_writer.as_default():
                tf.summary.scalar('actor_loss', actor_loss, step=episode)
                tf.summary.scalar('critic_loss', critic_loss, step=episode)
            # with graph_summary_writer.as_default():
            #     tf.summary.trace_export(name="update_target", step=episode,
            #                             profiler_outdir=graph_log_dir)

            state = next_state
            step += 1

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

