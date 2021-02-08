"""
PPO algorithm for Kuka Diverse Object Environment
Status: Work in Progress
- Best score is increasingly steadily. We can reach a score of 0.55 within 29 seasons.
- Implementation is based on information available at this link:
    https://towardsdatascience.com/proximal-policy-optimization-ppo-with-tensorflow-2-x-89c9430ecc26

"""
from pybullet_envs.bullet.kuka_diverse_object_gym_env import KukaDiverseObjectEnv
import numpy as np
import tensorflow as tf
from ppo2 import KukaPPOAgent
import datetime
import os
from collections import deque


#########################
# function definitions

# collect experiences for a certain number of episodes
def collect_trajectories(env, agent, tmax=128):

    states = []
    actions = []
    rewards = []
    dones = []
    obsv = env.reset()
    state = np.asarray(obsv, dtype=np.float32) / 255.0  # convert into float array
    for t in range(tmax):
        action = agent.policy(state)
        next_obsv, reward, done, _ = env.step(action)
        next_state = np.asarray(next_obsv, dtype=np.float32) / 255.0  # convert into float array
        dones.append(1 - done)
        rewards.append(reward)
        states.append(state)
        actions.append(action)
        state = next_state
        if done:
            env.reset()

    # compute an extra value required for GAE computation
    states.append(state)  # store one extra state

    return states, actions, rewards, dones


def main(env, agent):

    max_seasons = 1000
    tmax = 128
    max_epochs = 20
    path = './'

    filename = path + 'result_ppo_clip.txt'
    if os.path.exists(filename):
        os.remove(filename)
    else:
        print('The file does not exist. It will be created.')

    avg_rewards_list = []
    score_window = deque(maxlen=20)
    best_reward = 0
    for s in range(max_seasons):
        states, actions, rewards, dones = collect_trajectories(env, agent, tmax)

        # train
        a_loss = []
        c_loss = []
        for _ in range(max_epochs):
            al, cl = agent.train(states, actions, rewards, dones)
            a_loss.append(al)
            c_loss.append(cl)

        avg_reward = validate(env, agent, max_eps=20)
        avg_rewards_list.append(avg_reward)

        if avg_reward > best_reward:
            best_reward = avg_reward
            agent.save_model(path, 'actor_weights.h5', 'critic_weights.h5')
            print('Season:{}, Best Average Reward: {}. Model Saved'.format(
                s, avg_reward))

        with open(filename, 'a') as file:
            file.write('{}\t{:.2f}\t{:.2f}\t{:.2f}\n'.format(s, avg_reward, np.mean(a_loss),
                                                            np.mean(c_loss)))
        if s % 100 == 0:
            print('Season: {}'.format(s))
        if s > 25 and best_reward > 0.9:
            print('Problem is solved in seasons:{}'.format(s))
            break
    env.close()


# Test the model
def test(env, agent, path='./', max_eps=10 ):
    agent.load_model(path, 'actor_weights.h5', 'critic_weights.h5')

    input('Press Enter to continue')

    ep_reward_list = []
    for ep in range(max_eps):
        obsv = env.reset()
        state = np.asarray(obsv, dtype=np.float32) / 255.0  # convert into float array
        ep_reward = 0
        t = 0
        while True:
            env.render()        # show animation
            action = agent.policy(state)
            next_obsv, reward, done, _ = env.step(action)
            next_state = np.asarray(next_obsv, dtype=np.float32) / 255.0  # convert into float array
            ep_reward += reward
            t += 1
            state = next_state
            if done:
                ep_reward_list.append(ep_reward)
                print('Episode:{}, Reward:{}'.format(ep, ep_reward))
                break

    print('Avg Episodic Reward: ', np.mean(ep_reward_list))
    env.close()


# Validation Routine
def validate(env, agent, max_eps=20):
    ep_reward_list = []
    for ep in range(max_eps):
        obsv = env.reset()
        state = np.asarray(obsv, dtype=np.float32) / 255.0  # convert into float array
        t = 0
        ep_reward = 0
        while True:
            action = agent.policy(state)
            next_obsv, reward, done, _ = env.step(action)
            next_state = np.asarray(next_obsv, dtype=np.float32) / 255.0
            state = next_state
            ep_reward += reward
            t += 1
            if done:
                ep_reward_list.append(ep_reward)
                break

    mean_ep_reward = np.mean(ep_reward_list)
    return mean_ep_reward


####################
## main

if __name__ == '__main__':

    #####################
    # TENSORBOARD SETTINGS
    TB_LOG = False       # enable / disable tensorboard logging

    if TB_LOG:
        current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        train_log_dir = 'logs/train/' + current_time
        train_summary_writer = tf.summary.create_file_writer(train_log_dir)
    ############################
    # for reproducibility
    tf.random.set_seed(20)
    np.random.seed(20)

    #####################
    # start open/AI GYM environment
    env = KukaDiverseObjectEnv(renders=False,
                               isDiscrete=False,
                               maxSteps=20,
                               removeHeightHack=False)
    print('shape of Observation space: ', env.observation_space.shape)
    print('shape of Action space: ', env.action_space.shape)
    print('Reward Range: ', env.reward_range)
    print('Action High value: ', env.action_space.high)
    print('Action Low Value: ', env.action_space.low)


    ############################
    upper_bound = env.action_space.high
    state_size = env.observation_space.shape  # (48, 48, 3)
    action_size = env.action_space.shape  # (3,)
    print('state_size: ', state_size)
    print('action_size: ', action_size)

    # Create a Kuka Actor-Critic Agent
    agent = KukaPPOAgent(state_size, action_size, upper_bound,
                         lr_a=1e-4,
                         lr_c=2e-4,
                         gamma=0.99,
                         lmbda=0.7,
                         beta=0.5,
                         ent_coeff=0.01,
                         epsilon=0.05,
                         kl_target=0.01,
                         method='clip')
    # training
    main(env, agent)
    #test(env, agent)


