"""
PPO algorithm for Kuka Diverse Object Environment
Status: Not working. I did not wait till 150K iterations. I did not see any change during first 16K episodes.
"""
from pybullet_envs.bullet.kuka_diverse_object_gym_env import KukaDiverseObjectEnv
import numpy as np
import tensorflow as tf
from ppo import KukaPPOAgent
import datetime
from collections import deque


#########################
# function definitions
# collect experiences for a certain number of episodes
def collect_trajectories(env, agent, max_episodes):
    global train_summary_writer
    ep_reward_list = []
    steps = 0
    for ep in range(max_episodes):
        obsv = env.reset()
        state = np.asarray(obsv, dtype=np.float32) / 255.0  # convert into float array
        t = 0
        ep_reward = 0
        while True:
            action = agent.policy(state)
            next_obsv, reward, done, _ = env.step(action)
            next_state = np.asarray(next_obsv, dtype=np.float32) / 255.0  # convert into float array
            agent.record((state, action, reward, next_state, done))
            ep_reward += reward
            state = next_state
            t += 1

            if TB_LOG:          # tensorboard log
                tb_img = np.reshape(next_state, (-1,) + state_size)  # for tensorboard
                with train_summary_writer.as_default():
                    tf.summary.image("Training Image", tb_img, step=ep)
                    tf.summary.histogram("action_vector", action, step=t)

            if done:
                ep_reward_list.append(ep_reward)
                steps += t
                break

    mean_ep_reward = np.mean(ep_reward_list)
    return steps, mean_ep_reward

# main training routine
def main(env, agent, path='./'):
    global train_summary_writer

    if agent.method == 'clip':
        outfile = open(path + 'result_clip.txt', 'w')
    else:
        outfile = open(path + 'result_klp.txt', 'w')

    #training
    total_steps = 0
    best_score = -np.inf
    scores_window = deque(maxlen=100)
    for s in range(MAX_SEASONS):
        # collect trajectories
        t, s_reward = collect_trajectories(env, agent, TRG_EPISODES)

        # train the agent
        a_loss, c_loss, kld_value = agent.train(training_epochs=TRAIN_EPOCHS)

        # decay the clipping parameter over time
        agent.actor.epsilon *= 0.99

        scores_window.append(s_reward)

        total_steps += t
        print('Season: {}, Episodes: {} , Training Steps: {}, Mean Episodic Reward: {:.2f}'\
              .format(s, (s+1) * TRG_EPISODES, total_steps, s_reward))

        if TB_LOG:  # tensorboard logging
            with train_summary_writer.as_default():
                tf.summary.scalar('Mean reward', s_reward, step=s)
                tf.summary.scalar('Actor Loss', a_loss, step=s)
                tf.summary.scalar('Critic Loss', c_loss, step=s)
                tf.summary.scalar('KL Divergence', kld_value, step=s)
                tf.summary.scalar('beta', agent.actor.beta, step=s)

        #valid_score = validate(env, agent)
        mean_reward = np.mean(scores_window)
        if best_score < mean_reward:
            best_score = mean_reward
            agent.save_model(path, 'actor_weights.h5', 'critic_weights.h5')
            print('*** Season:{}, best score: {}. Model Saved ***'.format(s, best_score))

        # book keeping
        if agent.method == 'penalty':
            outfile.write('{}\t{:.2f}\t{:.2f}\t{:.2f}\t{:.2f}\t{:.2f}\n'.format(s,
                                    s_reward, a_loss, c_loss, kld_value, agent.actor.beta))
        else:
            outfile.write('{}\t{:.2f}\t{:.2f}\t{:.2f}\t{:.2f}\n'.format(s,
                                                        s_reward, a_loss, c_loss, kld_value))

        if s > 25 and best_score > 50:
            print('Problem is solved in {} seasons involving {} steps.'.format(s, total_steps))
            agent.save_model(path, 'actor_weights.h5', 'critic_weights.h5')
            break

    env.close()
    outfile.close()


# Test the model
def test(env, agent, path='./', max_eps=10):
    agent.load_model(path, 'actor_weights.h5', 'critic_weights.h5')

    ep_reward_list = []
    for ep in range(max_eps):
        obsv = env.reset()
        state = np.asarray(obsv, dtype=np.float32) / 255.0  # convert into float array
        ep_reward = 0
        t = 0
        while True:
            env.render()        # show animation
            action = agent.policy(state)
            next_obsv, reward, done, _ = agent.step(action)
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
    TB_LOG = True       # enable / disable tensorboard logging

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

    ################
    # Hyper-parameters
    ######################
    MAX_SEASONS = 30000
    TRG_EPISODES = 100      # training episodes in each season
    TEST_EPISODES = 20      # only for test
    TRAIN_EPOCHS = 20       # number of times agent is trained in each season
    BATCH_SIZE = 100
    MEMORY_CAPACITY = 10000     # memory capacity > trg_episodes * max_steps in each episode

    LR_A = 0.0001        # actor learning rate
    LR_C = 0.0002        # critic learning rate
    GAMMA = 0.99        # discount factor
    LAMBDA = 0.7        # required for GAE
    EPSILON = 0.02      # clipping factor
    BETA = 0.5          # required for 'Penalty' method
    ENT_COEFF = 0.01    # entropy coefficient
    KL_TARGET = 0.01
    METHOD = 'clip'     # choose 'clip' or 'penalty'

    ############################
    upper_bound = env.action_space.high
    state_size = env.observation_space.shape  # (48, 48, 3)
    action_size = env.action_space.shape  # (3,)
    print('state_size: ', state_size)
    print('action_size: ', action_size)

    # Create a Kuka Actor-Critic Agent
    agent = KukaPPOAgent(state_size, action_size, BATCH_SIZE,
                         MEMORY_CAPACITY, upper_bound,
                       LR_A, LR_C, GAMMA, LAMBDA, BETA, ENT_COEFF, EPSILON,
                         KL_TARGET, METHOD)
    # training
    main(env, agent)


