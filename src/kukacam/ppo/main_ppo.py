"""
PPO algorithm for Kuka Diverse Object Environment

- PPO Clip provides faster convergence compared to 'penalty' method. I have not tuned the parameters for 'penalty' method.
- Mean score over last 100 seasons increase over time with PPO clip.
- It is possible to reach mean episodic reward of about 0.7 - 0.8 over 20 episodes through the validation routine.

References:
https://github.com/mahyaret/kuka_rl/blob/master/kuka_rl_2.ipynb
https://towardsdatascience.com/proximal-policy-optimization-ppo-with-tensorflow-2-x-89c9430ecc26
"""
from pybullet_envs.bullet.kuka_diverse_object_gym_env import KukaDiverseObjectEnv
import numpy as np
import tensorflow as tf
from ppo import KukaPPOAgent
import datetime
from collections import deque
import os


#########################
# function definitions
# collect experiences for a certain number of episodes
def collect_trajectories(env, agent, tmax=1000):
    global train_summary_writer

    states = []
    next_states = []
    actions = []
    rewards = []
    dones = []
    ep_count = 0        # episode count

    obsv = env.reset()
    state = np.asarray(obsv, dtype=np.float32) / 255.0  # convert into float array
    for t in range(tmax):
        action = agent.policy(state)
        next_obsv, reward, done, _ = env.step(action)
        next_state = np.asarray(next_obsv, dtype=np.float32) / 255.0  # convert into float array

        states.append(state)
        actions.append(action)
        next_states.append(next_state)
        rewards.append(reward)
        dones.append(done)
        state = next_state

        if TB_LOG:          # tensorboard log
            tb_img = np.reshape(next_state, (-1,) + state_size)  # for tensorboard
            with train_summary_writer.as_default():
                tf.summary.image("Training Image", tb_img, step=t)
                tf.summary.histogram("action_vector", action, step=t)

        if done:
            ep_count += 1
            obsv = env.reset()
            state = np.asarray(obsv, dtype=np.float32) / 255.0

    return states, actions, rewards, next_states, dones, ep_count


# main training routine
def main(env, agent, path='./'):
    global train_summary_writer

    if agent.method == 'clip':
        filename = path + 'result_ppo_clip.txt'
    else:
        filename = path + 'result_ppo_klp.txt'

    if os.path.exists(filename):
        print('Deleting existing file. A new one will be created.')
        os.remove(filename)
    else:
        print('The file does not exist. It will be created.')

    #training
    max_seasons = 30
    best_score = 0
    # best_valid_score = 0
    scores_window = deque(maxlen=100)
    save_scores = []
    for s in range(max_seasons):
        # collect trajectories
        states, actions, rewards, next_states, dones, ep_count = \
            collect_trajectories(env, agent, tmax=1000)

        # train the agent
        a_loss, c_loss, kld_value = agent.train(states, actions, rewards,
                                                next_states, dones, epochs=10)

        # decay the clipping parameter over time
        agent.actor.epsilon *= 0.999
        agent.entropy_coeff *= 0.998

        season_score = np.sum(rewards, axis=0)
        scores_window.append(season_score)
        save_scores.append(season_score)
        mean_reward = np.mean(scores_window)
        # valid_score = validate(env, agent)

        print('Season: {}, season_score: {}, # episodes:{}, mean score:{:.2f}'\
              .format(s, season_score, ep_count, mean_reward))

        if TB_LOG:  # tensorboard logging
            with train_summary_writer.as_default():
                tf.summary.scalar('Mean reward', mean_reward, step=s)
                tf.summary.scalar('Actor Loss', a_loss, step=s)
                tf.summary.scalar('Critic Loss', c_loss, step=s)
                tf.summary.scalar('KL Divergence', kld_value, step=s)
                tf.summary.scalar('beta', agent.actor.beta, step=s)

        if best_score < mean_reward:
            best_score = mean_reward
            agent.save_model(path, 'actor_weights.h5', 'critic_weights.h5')
            print('*** Season:{}, best score: {}. Model Saved ***'.format(s, best_score))

        # if best_valid_score < valid_score:
        #     best_valid_score = valid_score
        #     agent.save_model(path, 'actor_wts_valid.h5', 'critic_wts_valid.h5')
        #     print('*** Season: {}, best validation score: {}. Model Saved ***'.format(s, best_valid_score))

        # book keeping
        if agent.method == 'penalty':
            with open(filename, 'a') as file:
                file.write('{}\t{:.2f}\t{:.2f}\t{:.2f}\t{:.2f}\t{:.2f}\t{:.2f}\n'.format(s,
                                    season_score, mean_reward, a_loss, c_loss, kld_value, agent.actor.beta))
        else:
            with open(filename, 'a') as file:
                file.write('{}\t{:.2f}\t{:.2f}\t{:.2f}\t{:.2f}\t{:.2f}\n'.format(s,
                                                        season_score, mean_reward, a_loss, c_loss, kld_value))

        if s > 25 and best_score > 50:
            print('Problem is solved in {} seasons.'.format(s))
            agent.save_model(path, 'actor_weights.h5', 'critic_weights.h5')
            break

    env.close()


# Test the model
def test(env, agent, path='./', max_eps=10):
    agent.load_model(path, 'actor_weights.h5', 'critic_weights.h5')
    # agent.load_model(path, 'actor_wts_valid.h5', 'critic_wts_valid.h5')

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
## main #####

if __name__ == '__main__':

    #####################
    # TENSORBOARD SETTINGS
    TB_LOG = False       # enable / disable tensorboard logging

    if TB_LOG:
        current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        train_log_dir = '../logs/train/' + current_time
        train_summary_writer = tf.summary.create_file_writer(train_log_dir)
    ############################
    # for reproducibility
    tf.random.set_seed(20)
    np.random.seed(20)

    #####################
    # start open/AI GYM environment
    env = KukaDiverseObjectEnv(renders=False,   # True for testing
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
    agent = KukaPPOAgent(state_size, action_size,
                         batch_size=128,
                         upper_bound=upper_bound,
                         lr_a=2e-4,
                         lr_c=2e-4,
                         gamma=0.99,
                         lmbda=0.95,
                         beta=0.5,
                         ent_coeff=0.01,
                         epsilon=0.2,
                         kl_target=0.01,
                         method='clip')
    # training
    main(env, agent)

    # testing
    # test(env, agent)


