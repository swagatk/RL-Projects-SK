"""
Here the run() function is separated from the class SACAgent2()
This will allow users to handle different types of actions and observations
"""
import gym
import numpy as np
import tensorflow as tf
import datetime
from collections import deque
import sys

sys.path.append(r'/content/gdrive/MyDrive/Colab/RL-Projects-SK/src/kukacam/')


# local imports
from SAC.sac2 import SACAgent2
from common.utils import uniquify

######################
# HYPERPARAMETERS
######################
SEASONS = 100
success_value = None
learning_rate = 0.0002
epochs = 20
buffer_capacity = 50000  # 50k (racecar)  # 20K (kuka)
training_episodes = 10000
batch_size = 128  # 512 (racecar) #   28 (kuka)
gamma = 0.993  # 0.99
lmbda = 0.7  # 0.9
tau = 0.995     # polyak averaging factor
alpha = 0.2     # Entropy Coefficient
use_attention = False  # enable/disable for attention model
use_mujoco = False
TB_LOG = True
tb_log_path = '/content/gdrive/MyDrive/Colab/tb_log/'
save_path = '/content/gdrive/MyDrive/Colab/kuka/sac/'

#################
# Functions


# validate function
def validate(env, agent, max_eps=50):
    ep_reward_list = []
    for ep in range(max_eps):
        state = env.reset()
        t = 0
        ep_reward = 0
        while True:
            action, _ = agent.policy(state)
            next_state, reward, done, _ = env.step(action)
            state = next_state
            ep_reward += reward
            t += 1
            if done:
                ep_reward_list.append(ep_reward)
                break
    # outside for loop
    mean_ep_reward = np.mean(ep_reward_list)
    return mean_ep_reward


# Main training function
def run(env, agent, training_episodes, success_value, filename, val_freq, path):
    #######################
    # TENSORBOARD SETTINGS
    if TB_LOG:
        current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        train_log_dir = tb_log_path + current_time
        train_summary_writer = tf.summary.create_file_writer(train_log_dir)
    ########################################

    if filename is None:
        filename = 'sac_output.txt'
    filename = uniquify(save_path + filename)

    if val_freq is not None:
        val_scores = deque(maxlen=50)
        val_score = 0

    start = datetime.datetime.now()
    best_score = -np.inf
    ep_lens = []  # episodic length
    ep_scores = []  # All episodic scores
    time_steps = 0
    for ep in range(training_episodes):
        # initial state
        state = env.reset()
        ep_score = 0  # score for each episode
        t = 0  # length of each episode
        while True:
            action, _ = agent.policy(state)
            action2 = np.reshape(action, action_size)
            next_state, reward, done, _ = env.step(action2)

            # store in replay buffer for off-policy training
            agent.buffer.record((state, action, reward, next_state, done))

            state = next_state
            ep_score += reward
            t += 1

            if done:
                time_steps += t
                break
            # done block ends here
        # end of one episode
        # off-policy training after each episode
        c1_loss, c2_loss, actor_loss, alpha_loss = agent.replay()
        ep_scores.append(ep_score)
        ep_lens.append(t)
        mean_ep_score = np.mean(ep_scores)
        mean_ep_len = np.mean(ep_lens)

        if ep > 100 and mean_ep_score > best_score:
            agent.save_model(save_path, 'actor_wts.h5', 'c1_wts.h5', 'c2_wts.h5', 'c1t_wts.h5', 'c2t_wts.h5')
            print('Episode: {}, Update best score: {}-->{}, Model saved!'.format(ep, best_score, mean_ep_score))
            best_score = mean_ep_score

        if val_freq is not None:
            if ep % val_freq == 0:
                print('Episode: {}, Score: {}, Mean score: {}'.format(ep, ep_score, mean_ep_score))
                val_score = validate(env)
                val_scores.append(val_score)
                mean_val_score = np.mean(val_scores)
                print('Episode: {}, Validation Score: {}, Mean Validation Score: {}' \
                      .format(ep, val_score, mean_val_score))

        if TB_LOG:
            with train_summary_writer.as_default():
                tf.summary.scalar('1. Episodic Score', ep_score, step=ep)
                tf.summary.scalar('2. Mean Season Score', mean_ep_score, step=ep)
                if val_freq is not None:
                    tf.summary.scalar('3. Validation Score', val_score, step=ep)
                tf.summary.scalar('4. Actor Loss', actor_loss, step=ep)
                tf.summary.scalar('5. Critic1 Loss', c1_loss, step=ep)
                tf.summary.scalar('6. Critic2 Loss', c2_loss, step=ep)
                tf.summary.scalar('7. Mean Episode Length', mean_ep_len, step=ep)
                tf.summary.scalar('8. Alpha Loss', alpha_loss, step=ep)

        with open(filename, 'a') as file:
            file.write('{}\t{}\t{:.2f}\t{:.2f}\t{:.2f}\t{:.2f}\t{:.2f}\t{:.2f}\t{:.2f}\n'
                       .format(ep, time_steps, mean_ep_len,
                               ep_score, mean_ep_score, actor_loss, c1_loss, c2_loss, alpha_loss))

        if success_value is not None:
            if best_score > success_value:
                print('Problem is solved in {} episodes with score {}'.format(ep, best_score))
                print('Mean Episodic score: {}'.format(mean_ep_score))
                break
    # end of episode-loop
    end = datetime.datetime.now()
    print('Time to Completion: {}'.format(end - start))
    env.close()
    print('Mean episodic score over {} episodes: {:.2f}'.format(training_episodes, np.mean(ep_scores)))


################
if __name__ == '__main__':
    env = gym.make('MountainCarContinuous-v0')
    state_size = env.observation_space.shape
    action_size = env.action_space.shape
    action_upper_bound = env.action_space.high[0]

    agent = SACAgent2(state_size, action_size, action_upper_bound, epochs,
                      batch_size, buffer_capacity, learning_rate, 
                 gamma, tau, alpha, use_attention, path=save_path)

    # train
    run(env, agent, training_episodes, success_value, filename='mc_sac.txt', val_freq=None, path=save_path)

