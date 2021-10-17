"""
Main script file for comparing performance of different algorithms
for the KukaDiverseObject Gym Environment. 
Algorithms being compared are: PPO, SAC, IPG, IPG+HER, SAC+HER
It will eventually supercede `main.py`

Differences compared to 'main.py': 
- the `run` function is separated from the RL-Agent. This will allow users to pre-process
    the inputs/ outputs before passing them to Agents. 

- It is integrated with `weights & Biases` for visualization

- The environments that will be tried:
1. KukaDiverseObject
2. KukaGrasp
3. RaceCar

To-DO list:
- Create an array to store the successful terminal states (for which reward = 1) and use it as hind_goal
- Does it affect the training performance if we use output of feature network while computing the her_reward.
- Does attention affect HER performance? If yes, how?
- visualize gradients for attention
- Incorporate LSTM into the model and analyze its effect
"""
# Import
from logging import currentframe
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
import os
from tensorflow.python.keras.backend import dtype
import wandb
import sys

# Add the current folder to python's import path
sys.path.append(os.path.dirname(os.path.realpath(__file__)))

# Local imports
from PPO.ppo2 import PPOAgent2
from IPG.ipg import IPGAgent2
from IPG.ipg_her import IPGHERAgent2
from SAC.sac2 import SACAgent2          # critic.train2()
from SAC.sac_her import SACHERAgent2
# from SAC.sac3 import SACAgent2        # critic.train()
from common.TimeLimitWrapper import TimeLimitWrapper
from common.CustomGymWrapper import ObsvnResizeTimeLimitWrapper
from common.utils import uniquify

########################################
# check versions
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
    buffer_capacity = 100000,    # 50k (racecar)  # 20K (kuka)
    batch_size = 256,  # 512 (racecar) #   128 (kuka)
    epsilon = 0.2,  # 0.07      # Clip factor required in PPO
    gamma = 0.993,  # 0.99      # discounted factor
    lmbda = 0.7,  # 0.9         # required for GAE in PPO
    tau = 0.995,                # polyak averaging factor
    alpha = 0.2,                # Entropy Coefficient   required in SAC
    use_attention = False,      # enable/disable attention model
    algo = 'sac_her',               # choices: ppo, sac, ipg, sac_her, ipg_her
)
####################################3
#  Additional Hyper-parameters
use_HER = True             # enable/disable HER
seasons = 35 
COLAB = False
env_name = 'kuka'
val_freq = None
WB_LOG = True
success_value = None 
LOG_FILE = True
load_path = None

#save_path = '/content/gdrive/MyDrive/Colab/kuka/sac/'
current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
save_path = './log/' + current_time + '/'

#######################################33
# wandb related configuration
if WB_LOG:
    print("WandB version", wandb.__version__)
    wandb.login()
    wandb.init(project='kukacam', config=config_dict)
#########################################
############################
# Google Colab Settings
if COLAB:
    import pybullet as p
    p.connect(p.DIRECT)
#################################3
###############################333
# Functions
################################
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

    if load_path is not None:
        agent.load_model(load_path)
        print('---Model parameters are loaded---')

    # create folder for storing result files
    if LOG_FILE:
        os.makedirs(save_path, exist_ok=True)
        tag = '_her' if use_HER else ''
        filename = env_name + '_' + config_dict['algo'] + tag + '.txt'
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
    s_scores = []       # season scores
    total_ep_cnt = 0  # total episode count
    global_time_steps = 0   # global training step counts
    for s in range(seasons):
        states, next_states, actions, rewards, dones = [], [], [], [], []

        if use_HER:
            goals = []
            temp_experience = []      # temporary experience buffer

        s_score = 0     # season score
        ep_cnt = 0      # no. of episodes in each season
        ep_steps = 0    # no. of steps in each episode
        ep_score = 0    # episodic reward
        done = False
        for t in range(config_dict['training_batch']):
            if use_HER:
                action, _ = agent.policy(state, goal)
            else:
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
                agent.buffer.record([state, action, reward, next_state, done, goal]) # check this
                # Also store in a separate temporary buffer
                temp_experience.append([state, action, reward, next_state, done, goal])
            else:
                agent.buffer.record([state, action, reward, next_state, done])

            state = next_state
            ep_score += reward
            ep_steps += 1       
            global_time_steps += 1

            if done:
                s_score += ep_score
                ep_cnt += 1         # episode count in each season
                total_ep_cnt += 1   # global episode count
                ep_scores.append(ep_score)
                ep_lens.append(ep_steps)

                if use_HER:
                    hind_goal = temp_experience[-1][3]  # Final state strategy
                    # add hindsight experience to the main buffer
                    agent.add_her_experience(temp_experience, hind_goal)
                    temp_experience = []    # clear the temporary buffer

                # off-policy training after each episode
                if config_dict['algo'] == 'sac_her':
                    a_loss, c_loss, alpha_loss = agent.train()
                    if WB_LOG:
                        wandb.log({'time_steps' : global_time_steps,
                            'Episodes' : total_ep_cnt,
                            'mean_ep_score': np.mean(ep_scores),
                            'ep_actor_loss' : a_loss,
                            'ep_critic_loss' : c_loss,
                            'ep_alpha_loss' : alpha_loss,
                            'mean_ep_len' : np.mean(ep_lens)},
                            step = total_ep_cnt)

                # prepare for next episode
                state = np.asarray(env.reset(), dtype=np.float32) / 255.0
                if use_HER: 
                    goal = np.asarray(env.reset(), dtype=np.float32) / 255.0
                ep_steps, ep_score = 0, 0
                done = False

            # done block ends here
        # end of one season

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
                if WB_LOG:
                    wandb.log({'val_score': val_score, 
                                'mean_val_score': val_score})

        if WB_LOG:
            wandb.log({'Season Score' : s_score, 
                        'Mean Season Score' : mean_s_score,
                        'Actor Loss' : a_loss,
                        'Critic Loss' : c_loss,
                        'Mean episode length' : mean_ep_len,
                        'Season' : s})

        if LOG_FILE:
            if config_dict['algo'] == 'sac_her':
                with open(filename, 'a') as file:
                    file.write('{}\t{}\t{}\t{:.2f}\t{:.2f}\t{:.2f}\t{:.2f}\t{:.2f}\t{:.2f}\n'
                            .format(s, total_ep_cnt, global_time_steps, mean_ep_len,
                                    s_score, mean_s_score, a_loss, c_loss, alpha_loss))
            elif config_dict['algo'] == 'ipg':
                with open(filename, 'a') as file:
                    file.write('{}\t{}\t{}\t{:.2f}\t{:.2f}\t{:.2f}\t{:.2f}\t{:.2f}\n'
                            .format(s, total_ep_cnt, global_time_steps, mean_ep_len,
                                    s_score, mean_s_score, a_loss, c_loss))

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


    # RL Agent
    if config_dict['algo'] == 'sac':
        agent = SACAgent2(state_size, action_size, action_upper_bound, 
                            config_dict['epochs'],
                            config_dict['batch_size'],
                            config_dict['buffer_capacity'],
                            config_dict['lr_a'],
                            config_dict['lr_c'],
                            config_dict['gamma'],
                            config_dict['tau'],
                            config_dict['alpha'],
                            config_dict['use_attention'])
    elif config_dict['algo'] == 'sac_her':
        agent = SACHERAgent2(state_size, action_size, action_upper_bound, 
                            config_dict['epochs'],
                            config_dict['batch_size'],
                            config_dict['buffer_capacity'],
                            config_dict['lr_a'],
                            config_dict['lr_c'],
                            config_dict['gamma'],
                            config_dict['tau'],
                            config_dict['alpha'],
                            config_dict['use_attention'])

    elif config_dict['algo'] == 'ipg':
        agent = IPGAgent2(state_size, action_size, action_upper_bound, 
                            config_dict['epochs'],
                            config_dict['batch_size'],
                            config_dict['buffer_capacity'],
                            config_dict['lr_a'],
                            config_dict['lr_c'],
                            config_dict['gamma'],
                            config_dict['tau'],
                            config_dict['alpha'],
                            config_dict['use_attention'])
    elif config_dict['algo'] == 'ipg_her':
        agent = IPGHERAgent2(state_size, action_size, action_upper_bound, 
                            config_dict['epochs'],
                            config_dict['batch_size'],
                            config_dict['buffer_capacity'],
                            config_dict['lr_a'],
                            config_dict['lr_c'],
                            config_dict['gamma'],
                            config_dict['tau'],
                            config_dict['alpha'],
                            config_dict['use_attention'])
    elif config_dict['algo'] == 'ppo':
        agent = PPOAgent2(state_size, action_size, action_upper_bound, 
                            config_dict['epochs'],
                            config_dict['batch_size'],
                            config_dict['buffer_capacity'],
                            config_dict['lr_a'],
                            config_dict['lr_c'],
                            config_dict['gamma'],
                            config_dict['tau'],
                            config_dict['alpha'],
                            config_dict['use_attention'])
    else:
        raise ValueError("invalid choice for algo. Exiting ...") 


    # Run / Train
    run(env, agent)