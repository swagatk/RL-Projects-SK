""" main for running ppo and ipg agents """

# Imports
from numpy.lib.npyio import save
import tensorflow as tf
from pybullet_envs.bullet.kuka_diverse_object_gym_env import KukaDiverseObjectEnv
from pybullet_envs.bullet.kukaCamGymEnv import KukaCamGymEnv
from pybullet_envs.bullet.racecarZEDGymEnv import RacecarZEDGymEnv
from pybullet_envs.bullet.racecarGymEnv import RacecarGymEnv
from packaging import version
import gym
import os
import datetime

# Add the current folder to python's import path
import sys
sys.path.append(os.path.dirname(os.path.realpath(__file__)))

# Local imports
from PPO.ppo_lstm import PPOAgent
# from PPO.ppo2 import PPOAgent
from IPG.ipg import IPGAgent
from IPG.ipg_her import IPGHERAgent
from SAC.sac import SACAgent
from SAC.sac_her import SACHERAgent
from common.TimeLimitWrapper import TimeLimitWrapper
from common.CustomGymWrapper import ObsvnResizeTimeLimitWrapper

########################################
# check tensorflow version
print("Tensorflow Version: ", tf.__version__)
assert version.parse(tf.__version__).release[0] >= 2, \
    "This program requires Tensorflow 2.0 or above"
#######################################

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
##########################################
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
    algo = 'ppo',               # choices: ppo, sac, ipg, sac_her, ipg_her
    env_name = 'kuka',          # environment name
)

####################################3
#  Additional parameters 
#########################################3
seasons = 35 
COLAB = False
WB_LOG = False
success_value = None 
############################
# Google Colab Settings
if COLAB:
    import pybullet as p
    p.connect(p.DIRECT)
    save_path = '/content/gdrive/MyDrive/Colab/' 
    CHKPT = True
    load_path = None
else:
    save_path = './log/'
    CHKPT = False
    load_path = None
##############################################3
save_path = save_path + config_dict['env_name'] + '/' + config_dict['algo'] + '/'
#current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
current_time = datetime.datetime.now().strftime("%Y%m%d")
save_path = save_path + current_time + '/'
logfile = config_dict['env_name'] + '_' + config_dict['algo'] + '.txt'
###########################################
# wandb related configuration
import wandb
if WB_LOG:
    print("WandB version", wandb.__version__)
    wandb.login()
    wandb.init(project='kukacam', config=config_dict)
#######################################################33

#################################3
if __name__ == "__main__":

    if config_dict['env_name'] == 'kuka':
        # Instantiate Gym Environment
        env = KukaDiverseObjectEnv(renders=False,
                                isDiscrete=False,
                                maxSteps=20,
                                removeHeightHack=False)

    # Select RL Agent
    if config_dict['algo'] == 'ppo':
        agent = PPOAgent(env, seasons, success_value, 
                            config_dict['epochs'],
                            config_dict['training_batch'],
                            config_dict['batch_size'],
                            config_dict['lr_a'],
                            config_dict['lr_c'],
                            config_dict['gamma'],
                            config_dict['epsilon'],
                            config_dict['lmbda'],
                            config_dict['use_attention'],
                            filename=logfile, 
                            wb_log=WB_LOG,  
                            chkpt=CHKPT,
                            path=save_path)
    elif config_dict['algo'] == 'ipg':
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
                            filename=logfile, 
                            wb_log=WB_LOG,  
                            chkpt=CHKPT,
                            path=save_path)
    elif config_dict['algo'] == 'ipg_her':
        agent = IPGHERAgent(env, seasons, success_value, 
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
                            filename=logfile, 
                            wb_log=WB_LOG,  
                            chkpt=CHKPT,
                            path=save_path)
    elif config_dict['algo'] == 'sac':
        agent = SACAgent(env, seasons, success_value,
                            config_dict['epochs'],
                            config_dict['training_batch'],
                            config_dict['batch_size'],
                            config_dict['buffer_capacity'],
                            config_dict['lr_a'],
                            config_dict['lr_c'],
                            config_dict['gamma'],
                            config_dict['tau'],
                            config_dict['alpha'],
                            config_dict['use_attention'],
                            filename=logfile, 
                            wb_log=WB_LOG,  
                            chkpt=CHKPT,
                            path=save_path)
    elif config_dict['algo'] == 'sac_her':
        agent = SACHERAgent(env, seasons, success_value,
                            config_dict['epochs'],
                            config_dict['training_batch'],
                            config_dict['batch_size'],
                            config_dict['buffer_capacity'],
                            config_dict['lr_a'],
                            config_dict['lr_c'],
                            config_dict['gamma'],
                            config_dict['tau'],
                            config_dict['alpha'],
                            config_dict['use_attention'],
                            filename=logfile, 
                            wb_log=WB_LOG,  
                            chkpt=CHKPT,
                            path=save_path)
    else:
        raise ValueError('Invalid Choice of Algorithm. Exiting ...')

    # Train
    agent.run()