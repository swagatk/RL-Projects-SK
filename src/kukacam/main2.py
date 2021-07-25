"""
Main script file for comparing performance of different algorithms
for the KukaDiverseObject Gym Environment. 
Algorithms being compared are: PPO, SAC, IPG, IPG+HER, SAC+HER

- the `run` function is separated from the RL-Agent. This will allow users to pre-process
    the inputs/ outputs before passing them to Agents. 

- The environments that will be tried:
1. KukaDiverseObject
2. KukaGrasp
3. RaceCar
"""
# Imports
import tensorflow as tf
from pybullet_envs.bullet.kuka_diverse_object_gym_env import KukaDiverseObjectEnv
from pybullet_envs.bullet.kukaCamGymEnv import KukaCamGymEnv
from pybullet_envs.bullet.racecarZEDGymEnv import RacecarZEDGymEnv
from pybullet_envs.bullet.racecarGymEnv import RacecarGymEnv
from packaging import version
from collections import deque
from datetime import datetime
import numpy as np
import gym
import sys
import wandb

# Add the current folder to python's import path
sys.path.append('/home/swagat/GIT/RL-Projects-SK/src/kukacam/')

# Local imports
from ppo.ppo2 import PPOAgent
from IPG.ipg import IPGAgent
from IPG.ipg_her import IPGHERAgent
from SAC.sac2 import SACAgent2
from common.TimeLimitWrapper import TimeLimitWrapper
from common.CustomGymWrapper import ObsvnResizeTimeLimitWrapper
from common.utils import uniquify


########################################
# check tensorflow version
print("Tensorflow Version: ", tf.__version__)
assert version.parse(tf.__version__).release[0] >= 2, \
    "This program requires Tensorflow 2.0 or above"
######################################
# avoid CUDNN_STATUS_INTERNAL_ERROR
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
config.log_device_placement = True
sess = tf.compat.v1.Session(config=config)
################################################
# check GPU device
device_name = tf.test.gpu_device_name()
if device_name != '/device:GPU:0':
    raise SystemError('GPU device not found')
print('Found GPU at: {}'.format(device_name))
##############################################
# wandb related configuration
wandb.login()
wandb.init(project='kukacam')
#########################################
# #### Hyper-parameters
##############################
SEASONS = 50   # 35
success_value = None
lr_a = 0.0002  
lr_c = 0.0002  
epochs = 50
training_batch = 1024   # 5120(racecar)  # 1024 (kuka), 512
training_episodes = 10000    # needed for SAC
buffer_capacity = 100000     # 50k (racecar)  # 20K (kuka)
batch_size = 256   # 512 (racecar) #   28 (kuka)
epsilon = 0.2  # 0.07
gamma = 0.993  # 0.99
lmbda = 0.7  # 0.9
tau = 0.995             # polyak averaging factor
alpha = 0.2             # Entropy Coefficient
use_attention = False  # enable/disable for attention model
use_mujoco = False
TB_LOG = True
save_path = './log/'
algo = 'sac'

#################################3
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
def run(env, agent, training_episodes, success_value, filename, val_freq):
    #######################
    # TENSORBOARD SETTINGS
    if TB_LOG:
        current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        train_log_dir = save_path + 'tb_log/' + current_time
        wandb.tensorboard.patch(root_logdir=train_log_dir)   
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
        obs = env.reset()
        state = np.asarray(obs, dtype=np.float32) / 255.0
        ep_score = 0  # score for each episode
        t = 0  # length of each episode
        while True:
            action, _ = agent.policy(state)
            next_obs, reward, done, _ = env.step(action)
            next_state = np.asarray(next_obs, dtype=np.float32) / 255.0

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
        if algo == 'sac':
            c1_loss, c2_loss, actor_loss, alpha_loss = agent.replay()
        elif algo == 'ppo':
            actor_loss, critic_loss = agent.replay()
        else:
            actor_loss, critic_loss = agent.replay()

        ep_scores.append(ep_score)
        ep_lens.append(t)
        mean_ep_score = np.mean(ep_scores)
        mean_ep_len = np.mean(ep_lens)

        if ep > 100 and mean_ep_score > best_score:
            agent.save_model(save_path)
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

        if algo == 'sac':
            with open(filename, 'a') as file:
                file.write('{}\t{}\t{:.2f}\t{:.2f}\t{:.2f}\t{:.2f}\t{:.2f}\t{:.2f}\t{:.2f}\n'
                        .format(ep, time_steps, mean_ep_len,
                                ep_score, mean_ep_score, actor_loss, c1_loss, c2_loss, alpha_loss))
        elif algo == 'ppo':
            pass
        elif algo == 'ipg':
            pass
        elif algo == 'ipg_her':
            pass
        else:
            raise ValueError("Invalid Choice of Algorithm. Select sac/ppo/ipg/ipg_her")

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

if __name__ == "__main__":

    # Kuka DiverseObject Environment
    env = KukaDiverseObjectEnv(renders=False,
                               isDiscrete=False,
                               maxSteps=20,
                               removeHeightHack=False)

    state_size = env.observation_space.shape
    action_size = env.action_space.shape
    action_upper_bound = env.action_space.high

    # RaceCar Bullet Environment with image observation
    # env = ObsvnResizeTimeLimitWrapper(RacecarZEDGymEnv(renders=False,
    #                            isDiscrete=False), shape=20, max_steps=20)

    # RaceCar Bullet Environment with vector observation
    # env = RacecarGymEnv(renders=False, isDiscrete=False)

    # PPO Agent
    # agent = PPOAgent(env, SEASONS, success_value, lr_a, lr_c, epochs, training_batch, batch_size, epsilon, gamma,
    #                  lmbda, use_attention, use_mujoco,
    #                  filename='rc_ppo_zed.txt', val_freq=None)
    # IPG Agent
    # agent = IPGAgent(env, SEASONS, success_value, lr_a, lr_c, epochs, training_batch, batch_size, buffer_capacity,
    #                  epsilon, gamma, lmbda, use_attention, use_mujoco,
    #                  filename='rc_ipg_zed.txt', val_freq=None)
    # IPG HER Agent
    # agent = IPGHERAgent(env, SEASONS, success_value, lr_a, lr_c, epochs, training_batch, batch_size,
    #                     buffer_capacity, epsilon, gamma, lmbda, use_attention,
    #                     use_mujoco, filename='rc_ipg_her.txt', val_freq=None)

    # SAC Agent
    # agent = SACAgent(env, success_value,
    #                  epochs, training_episodes, batch_size, buffer_capacity, lr_a, lr_c, alpha,
    #                  gamma, tau, use_attention, use_mujoco,
    #                  filename='kuka_sac.txt', tb_log=False, val_freq=None)

    
    agent = SACAgent2(state_size, action_size, action_upper_bound, epochs,
                      batch_size, buffer_capacity, learning_rate, 
                 gamma, tau, alpha, use_attention, path=save_path)

    agent.run()