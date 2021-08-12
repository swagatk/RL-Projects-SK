"""
Visualize features using tSNE plot, before and after training
"""
import tensorflow as tf
import numpy as np
import gym
import os
from pybullet_envs.bullet.kuka_diverse_object_gym_env import KukaDiverseObjectEnv
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns

# Add the current folder to python's import path
import sys
current_directory = os.path.dirname(os.path.realpath(__file__))
sys.path.append(current_directory)
sys.path.append(os.path.dirname(current_directory))


# Local imports
from IPG.ipg import IPGAgent


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
    algo = 'ipg',               # choices: ppo, sac, ipg, sac_her, ipg_her
    env_name = 'kuka',          # environment name
    her_strategy = 'future'     # HER strategy: final, future, success 
)

seasons = 35 
WB_LOG = False
success_value = None 
chkpt_freq = None
save_path = './logimg/'
log_file = None
load_path = '/home/swagat/GIT/RL-Projects-SK/src/kukacam/log/kuka/ipg/20210807/best_model/'

# Environment
env = KukaDiverseObjectEnv(renders=False,
                        isDiscrete=False,
                        maxSteps=20,
                        removeHeightHack=False)

# RL Agent
agent = IPGAgent(env, seasons, success_value,
                 config_dict['epochs'],
                 config_dict['training_batch'],
                 config_dict['batch_size'],
                 config_dict['buffer_capacity'],
                 config_dict['lr_a'],
                 config_dict['lr_c'],
                 config_dict['gamma'],
                 config_dict['epsilon'],
                 config_dict['lmbda'],
                 config_dict['use_attention'],
                 filename=log_file,
                 wb_log=WB_LOG,
                 chkpt_freq=chkpt_freq,
                 path=save_path)

# load model parameters
agent.load_model(load_path)


feature_list = []
action_list = []
reward_list = []
ep_reward_list = []
for ep in range(50):
    obs = env.reset()
    state = np.asarray(obs, dtype=np.float32) / 255.0
    t = 0
    ep_reward = 0
    while True:
        f = agent.extract_feature(state)
        action = agent.policy(state)
        next_obs, reward, done, _ = env.step(action)
        next_state = np.asarray(next_obs, dtype=np.float32) / 255.0
        
        feature_list.append(f)
        action_list.append(action)
        reward_list.append(reward)

        img_name = save_path + 'img_{}.png'.format(t)
        plt.savefig(img_name, format='png')

        state = next_state
        ep_reward += reward
        t += 1

        if done:
            ep_reward_list.append(ep_reward)
            break

print('Mean ep reward over {} episodes:{}'.format(ep, np.mean(ep_reward_list)))


#########
## TSNE plot

print('shape of feature list:', np.shape(np.array(feature_list)))

tsne = TSNE(n_components=2, n_iter=300, verbose=1)
tsne_result = tsne.fit_transform(feature_list)

sns.scatterplot(x=tsne_result[:,0], y = tsne_result[:,1])

