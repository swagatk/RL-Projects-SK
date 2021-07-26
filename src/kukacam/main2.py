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
# Import
import tensorflow as tf
from pybullet_envs.bullet.kuka_diverse_object_gym_env import KukaDiverseObjectEnv
from pybullet_envs.bullet.kukaCamGymEnv import KukaCamGymEnv
from pybullet_envs.bullet.racecarZEDGymEnv import RacecarZEDGymEnv
from pybullet_envs.bullet.racecarGymEnv import RacecarGymEnv
from packaging import version
from collections import deque
import datetime
import numpy as np
import gym
from tensorflow.python.keras.backend import dtype
import wandb

# Local imports
from ppo.ppo2 import PPOAgent
from IPG.ipg import IPGAgent
from IPG.ipg_her import IPGHERAgent
from SAC.sac2 import SACAgent2
from common.TimeLimitWrapper import TimeLimitWrapper
from common.CustomGymWrapper import ObsvnResizeTimeLimitWrapper
from common.utils import uniquify

# Add the current folder to python's import path
import sys
# sys.path.append('/content/gdrive/MyDrive/Colab/RL-Projects-SK/src/kukacam/')
sys.path.append('/home/swagat/GIT/RL-Projects-SK/src/kukacam/')



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
# #### Hyper-parameters
##############################
config_dict = dict(
    lr_a = 0.0002, 
    lr_c = 0.0002, 
    epochs = 20, 
    training_batch = 1024,    # 5120(racecar)  # 1024 (kuka), 512
    buffer_capacity = 20000,    # 50k (racecar)  # 20K (kuka)
    batch_size = 128,  # 512 (racecar) #   128 (kuka)
    epsilon = 0.2,  # 0.07      # Clip factor required in PPO
    gamma = 0.993,  # 0.99      # discounted factor
    lmbda = 0.7,  # 0.9         # required for GAE in PPO
    tau = 0.995,                # polyak averaging factor
    alpha = 0.2,                # Entropy Coefficient   required in SAC
    use_attention = False,      # enable/disable attention model
    algo = 'sac',
)

# Additional hyperparameters
use_HER = False             # enable/disable HER
seasons = 50 
COLAB = False
env_name = 'kuka'
val_freq = None
TB_LOG = True
success_value = None 
#save_path = '/content/gdrive/MyDrive/Colab/kuka/sac/'
save_path = './'
use_mujoco = False
#######################################33
# wandb related configuration
# wandb.login()
wandb.tensorboard.patch(root_logdir=save_path + 'tb_log/')   
wandb.init(project='kukacam', config=config_dict)
#########################################
############################
# Google Colab Settings
if COLAB:
    import pybullet as p
    p.connect(p.DIRECT)
#################################3
# TENSORBOARD SETTINGS
if TB_LOG:
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    train_log_dir = save_path + 'tb_log/' + current_time
    train_summary_writer = tf.summary.create_file_writer(train_log_dir)
########################################
###############################333
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
def run(env, agent):

    # file for storing results
    tag = '_her' if use_HER else ''
    filename = env_name + '_' + wandb.config.algo + tag + '.txt'
    filename = uniquify(save_path + filename)

    if val_freq is not None:
        val_scores = deque(maxlen=50)
        val_score = 0

    # initial state
    obs = env.reset()
    state = np.asarray(obs, dtype=np.float32) / 255.0
    if use_HER:
        goal = np.asarray(env.reset(), dtype=np.float32) / 255.0

    start = datetime.datetime.now()
    best_score = -np.inf
    ep_lens = []  # episodic length
    ep_scores = []      # All episodic scores
    time_steps = 0      # global step count
    s_scores = []       # season scores
    total_ep_cnt = 0  # total episode count
    for s in range(seasons):
        states, next_states, actions, rewards, dones = [], [], [], [], []

        if use_HER:
            goals = []
            temp_experience = []      # temporary experience buffer

        s_score = 0
        ep_cnt = 0      # no. of episodes in each season
        ep_score = 0    # episodic reward
        done = False
        for t in range(wandb.config.training_batch):
            action, _ = agent.policy(state)
            next_obs, reward, done, _ = env.step(action)
            next_state = np.asarray(next_obs, dtype=np.float32) / 255.0

            # this is used for on-policy training
            states.append(state)
            next_states.append(next_state)
            actions.append(action)
            rewards.append(reward)
            dones.append(done)

            # store in replay buffer for off-policy training
            if use_HER:
                goals.append(goal)
                agent.buffer.record((state, action, reward, next_state, done, goal)) # check this
                # Also store in a separate buffer
                temp_experience.append([state, action, reward, next_state, done, goal])
            else:
                agent.buffer.record((state, action, reward, next_state, done))


            state = next_state
            ep_score += reward
            t += 1

            if done:
                time_steps += t
                s_score += ep_score
                ep_cnt += 1         # episode count in each season
                total_ep_cnt += 1   # global episode count
                ep_scores.append(ep_score)
                ep_lens.append(t)

                if use_HER:
                    hind_goal = temp_experience[-1][3]  # Final state strategy
                    # add hindsight experience to the main buffer
                    agent.add_her_experience(temp_experience, hind_goal)
                    temp_experience = []    # clear the temporary buffer


                # off-policy training after each episode
                if wandb.config.algo == 'sac':
                    actor_loss, critic_loss, alpha_loss = agent.replay()
                    wandb.log({'ep_reward': ep_score,
                            'mean_ep_score': np.mean(ep_scores),
                            'ep_actor_loss': actor_loss,
                            'ep_critic_loss': critic_loss,
                            'ep_alpha_loss': alpha_loss},
                            step = total_ep_cnt)

                # prepare for next episode
                obs = env.reset()
                state = np.asarray(obs, dtype=np.float32) / 255.0
                if use_HER: 
                    state = np.asarray(env.reset(), dtype=np.float32) / 255.0
                ep_score = 0
                done = False

            # done block ends here
        # end of one season

        # off-policy training after each season 
        if wandb.config.algo == 'ppo':
            pass
        elif wandb.config.algo == 'ipg':
            if use_HER:
                a_loss, c_loss = agent.replay(states, actions, rewards, next_states, dones, goals)
            else:
                a_loss, c_loss = agent.replay(states, actions, rewards, next_states, dones)
        else:
            pass

        s_score = np.mean(ep_scores[-ep_cnt:])  # mean of last ep_cnt episodes
        s_scores.append(s_score)
        mean_s_score = np.mean(s_scores)
        mean_ep_score = np.mean(ep_scores)
        mean_ep_len = np.mean(ep_lens)

        if  mean_s_score > best_score:
            agent.save_model(save_path)
            print('Season: {}, Update best score: {}-->{}, Model saved!'.format(s, best_score, mean_s_score))
            best_score = mean_s_score

        if val_freq is not None:
            if total_ep_cnt % val_freq == 0:
                print('Episode: {}, Score: {}, Mean score: {}'.format(total_ep_cnt, ep_score, mean_ep_score))
                val_score = validate(env)
                val_scores.append(val_score)
                mean_val_score = np.mean(val_scores)
                print('Episode: {}, Validation Score: {}, Mean Validation Score: {}' \
                      .format(total_ep_cnt, val_score, mean_val_score))

        if TB_LOG:
            with train_summary_writer.as_default():
                tf.summary.scalar('1. Season Score', s_score, step=s)
                tf.summary.scalar('2. Mean season Score', mean_s_score, step=s)
                if val_freq is not None:
                    tf.summary.scalar('3. Validation Score', val_score, step=s)
                tf.summary.scalar('4. Actor Loss', actor_loss, step=s)
                tf.summary.scalar('5. Critic Loss', critic_loss, step=s)
                tf.summary.scalar('6. Mean Episode Length', mean_ep_len, step=s)
                if wandb.config.algo == 'sac':
                    tf.summary.scalar('7. Alpha Loss', alpha_loss, step=s)


        if wandb.config.algo == 'sac':
            with open(filename, 'a') as file:
                file.write('{}\t{}\t{}\t{:.2f}\t{:.2f}\t{:.2f}\t{:.2f}\t{:.2f}\t{:.2f}\n'
                        .format(s, total_ep_cnt, time_steps, mean_ep_len,
                                s_score, mean_s_score, actor_loss, critic_loss, alpha_loss))
        elif wandb.config.algo == 'ipg':
            with open(filename, 'a') as file:
                file.write('{}\t{}\t{}\t{:.2f}\t{:.2f}\t{:.2f}\t{:.2f}\t{:.2f}\n'
                        .format(s, total_ep_cnt, time_steps, mean_ep_len,
                                s_score, mean_s_score, actor_loss, critic_loss))

        if success_value is not None:
            if best_score > success_value:
                print('Problem is solved in {} seasons with best score {}'.format(s, best_score))
                print('Mean season score: {}'.format(mean_s_score))
                break
    # end of episode-loop
    end = datetime.datetime.now()
    print('Time to Completion: {}'.format(end - start))
    env.close()
    print('Mean episodic score over {} episodes: {:.2f}'.format(total_ep_cnt, np.mean(ep_scores)))

    if COLAB:
        p.disconnect(p.DIRECT) # on google colab
    # end of run function


######################################
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
    agent = SACAgent2(state_size, action_size, action_upper_bound, 
                        wandb.config.epochs,
                        wandb.config.batch_size, 
                        wandb.config.buffer_capacity, 
                        wandb.config.lr_a, 
                        wandb.config.lr_c, 
                        wandb.config.gamma, 
                        wandb.config.tau, 
                        wandb.config.alpha, 
                        wandb.config.use_attention)

    run(env, agent)