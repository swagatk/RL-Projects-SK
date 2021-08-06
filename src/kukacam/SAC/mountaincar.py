import gym
from SAC.sac import SACAgent

######################
# HYPERPARAMETERS
######################
SEASONS = 1000
success_value = None
lr_a = 0.0002  # 0.0002
lr_c = 0.0002  # 0.0002
epochs = 20
training_batch = 1000
buffer_capacity = 50000  # 50k (racecar)  # 20K (kuka)
batch_size = 128  # 512 (racecar) #   28 (kuka)
gamma = 0.993  # 0.99
lmbda = 0.7  # 0.9
tau = 0.995     # polyak averaging factor
alpha = 0.2     # Entropy Coefficient
use_attention = False  # enable/disable for attention model
use_mujoco = False

env = gym.make('MountainCarContinuous-v0')
agent = SACAgent(env, 100, success_value,
             epochs, training_batch, batch_size, buffer_capacity, 
             lr_a, lr_c, gamma, tau, alpha, use_attention, 
            filename='mc_sac.txt', wb_log=False, chkpt=False,
            path='./log/')



agent.run()


