import gym
from SAC.sac import SACAgent

######################
# HYPERPARAMETERS
######################
SEASONS = 100
success_value = None
lr_a = 0.0002  # 0.0002
lr_c = 0.0002  # 0.0002
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

env = gym.make('MountainCarContinuous-v0')


agent = SACAgent(env, success_value,
             epochs, training_episodes, batch_size, buffer_capacity, lr_a, lr_c, alpha,
             gamma, tau, use_attention, use_mujoco,
             filename='mc_sac.txt', tb_log=False, val_freq=None)



agent.run()


