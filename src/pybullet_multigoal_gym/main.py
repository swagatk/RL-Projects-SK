""" 
Main python file for implementing following algorithms:
- SAC
- SAC + HER
- PPO
- IPG
- IPG + HER

Environment: KUKADiverseObjectEnv

Updates:
18/08/2021: This is main file for kuka environment.
"""
# Imports
from numpy.lib.npyio import save
import tensorflow as tf
import pybullet_multigoal_gym as pmg
from packaging import version
import os
import datetime

# Add the current folder to python's import path
import sys
sys.path.append(os.path.dirname(os.path.realpath(__file__)))

# Local imports
from ppo import PPOAgent
from ipg import IPGAgent
#from ipg_her import IPGHERAgent
#from ipg_her_vae import IPGHERAgent
from ipg_her_pbmg import IPGHERAgent_pbmg
from sac import SACAgent
from sac_her import SACHERAgent

########################################
# check tensorflow version
print("Tensorflow Version: ", tf.__version__)
assert version.parse(tf.__version__).release[0] >= 2, \
    "This program requires Tensorflow 2.0 or above"
#######################################

######################################
# avoid CUDNN_STATUS_INTERNAL_ERROR
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            print(gpu)
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
# if device_name != '/device:GPU:0':
#     raise SystemError('GPU device not found')
print('Found GPU at: {}'.format(device_name))
##############################################
# #### Hyper-parameters for RACECAR Environment
##########################################
config_dict = dict(
    lr_a = 0.0002, 
    lr_c = 0.0002, 
    epochs = 20, 
    training_batch = 2560,    # 5120(racecar)  # 1024 (kuka), 512
    buffer_capacity = 30000,    # 50k (racecar)  # 20K (kuka)
    batch_size = 128,  # 512 (racecar) #   128 (kuka)
    epsilon = 0.2,  # 0.07      # Clip factor required in PPO
    gamma = 0.993,  # 0.99      # discounted factor
    lmbda = 0.7,  # 0.9         # required for GAE in PPO
    tau = 0.995,                # polyak averaging factor
    alpha = 0.2,                # Entropy Coefficient   required in SAC
    # use_attention = {'type': 'luong',   # type: luong, bahdanau
    #                  'arch': 0,         # arch: 0, 1, 2, 3
    #                  'return_scores': False},  # visualize attention maps       
    use_attention = None, 
    algo = 'ipg_her',               # choices: ppo, sac, ipg, sac_her, ipg_her
    env_name = 'pbmg',          # environment name
    use_her = { 'strategy': 'final',
                'extract_feature' : False}, 
    stack_size = 0,             # input stack size
    use_lstm = False,             # enable /disable LSTM
)
#######################
WB_LOG = True
COLAB = False
############################
# Google Colab Settings
if COLAB:
    import pybullet as p
    p.connect(p.DIRECT)
    save_path = '/content/gdrive/MyDrive/Colab/' 
    chkpt_freq = 5
    load_path = None
else:
    save_path = './log/'
    chkpt_freq = None         # wrt seasons
    load_path = None
##############################################3
save_path = save_path + config_dict['env_name'] + '/' + config_dict['algo'] + '/'
current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
#current_time = datetime.datetime.now().strftime("%Y%m%d")
save_path = save_path + current_time + '/'
#logfile = config_dict['env_name'] + '_' + config_dict['algo'] + '.txt'
logfile = None  # Don't write  into file
###########################################
# wandb related configuration
import wandb
if WB_LOG:
    #WANDB_API_KEY=os.environ['MY_WANDB_API_KEY']
    print("WandB version", wandb.__version__)
    wandb.login()
    wandb.init(project=config_dict['env_name'], config=config_dict)
#######################################################33
#################################3
if __name__ == "__main__":

    if config_dict['env_name'] == 'pbmg':

        camera_setup = [
            {
                'cameraEyePosition': [-1.0, 0.25, 0.6],
                'cameraTargetPosition': [-0.6, 0.05, 0.2],
                'cameraUpVector': [0, 0, 1],
                'render_width': 128,
                'render_height': 128
            },
            {
                'cameraEyePosition': [-1.0, -0.25, 0.6],
                'cameraTargetPosition': [-0.6, -0.05, 0.2],
                'cameraUpVector': [0, 0, 1],
                'render_width': 128,
                'render_height': 128
            }
        ]

        env = pmg.make_env(
            # task args ['reach', 'push', 'slide', 'pick_and_place', 
            #            'block_stack', 'block_rearrange', 'chest_pick_and_place', 'chest_push']
            task='reach',
            gripper='parallel_jaw',
            num_block=4,  # only meaningful for multi-block tasks
            render=False,
            binary_reward=True,
            max_episode_steps=5,
            # image observation args
            image_observation=False,
            depth_image=False,
            goal_image=True,
            visualize_target=True,
            camera_setup=camera_setup,
            observation_cam_id=0,
            goal_cam_id=1,
            # curriculum args
            use_curriculum=True,
            num_goals_to_generate=90)


    #print('Observation:', env.observation_space)
    image_observation = False
    if image_observation: 
        state_size = env.observation_space['observation'].__getattribute__('shape')
    else:
        state_size = env.observation_space['state'].__getattribute__('shape')

    action_size = env.action_space.__getattribute__('shape')
    upper_bound = env.action_space.__getattribute__('high')
    

    print('State size:', state_size)
    print('Action size:', action_size)
    print('Action upper bound:', upper_bound)

    ans = input('Do you want to continue?(Y/N)')
    if ans.lower() == 'n':
        exit()

    # Select RL Agent
    if config_dict['algo'] == 'ppo':
        pass
        # agent = PPOAgent(env=env, 
        #                 seasons=50, 
        #                 success_value=None, 
        #                     config_dict['epochs'],
        #                     config_dict['training_batch'],
        #                     config_dict['batch_size'],
        #                     config_dict['lr_a'],
        #                     config_dict['lr_c'],
        #                     config_dict['gamma'],
        #                     config_dict['epsilon'],
        #                     config_dict['lmbda'],
        #                     config_dict['use_attention'],
        #                     filename=logfile, 
        #                     wb_log=WB_LOG,  
        #                     chkpt_freq=chkpt_freq,
        #                     path=save_path)
    elif config_dict['algo'] == 'ipg':
        pass
        # agent = IPGAgent(env, seasons, success_value, 
        #                     config_dict['epochs'],
        #                     config_dict['training_batch'],
        #                     config_dict['batch_size'],
        #                     config_dict['buffer_capacity'],
        #                     config_dict['lr_a'],
        #                     config_dict['lr_c'],
        #                     config_dict['gamma'],
        #                     config_dict['epsilon'],
        #                     config_dict['lmbda'],
        #                     config_dict['stack_size'],
        #                     config_dict['use_attention'],
        #                     filename=logfile, 
        #                     wb_log=WB_LOG,  
        #                     chkpt_freq=chkpt_freq,
        #                     path=save_path)
    elif config_dict['algo'] == 'ipg_her':
        agent = IPGHERAgent_pbmg(
                            **config_dict,
                            seasons = 100,
                            success_value = None,
                            env = env,
                            state_size = state_size,
                            action_size = action_size,
                            upper_bound = upper_bound,
                            filename=logfile, 
                            wb_log=WB_LOG,
                            chkpt_freq=None,
                            path=save_path)
    elif config_dict['algo'] == 'sac':
        pass
        # agent = SACAgent(env, seasons, success_value,
        #                     config_dict['epochs'],
        #                     config_dict['training_batch'],
        #                     config_dict['batch_size'],
        #                     config_dict['buffer_capacity'],
        #                     config_dict['lr_a'],
        #                     config_dict['lr_c'],
        #                     config_dict['gamma'],
        #                     config_dict['tau'],
        #                     config_dict['alpha'],
        #                     config_dict['use_attention'],
        #                     filename=logfile, 
        #                     wb_log=WB_LOG,  
        #                     chkpt_freq=chkpt_freq,
        #                     path=save_path)
    elif config_dict['algo'] == 'sac_her':
        pass
        # agent = SACHERAgent(env, seasons, success_value,
        #                     config_dict['epochs'],
        #                     config_dict['training_batch'],
        #                     config_dict['batch_size'],
        #                     config_dict['buffer_capacity'],
        #                     config_dict['lr_a'],
        #                     config_dict['lr_c'],
        #                     config_dict['gamma'],
        #                     config_dict['tau'],
        #                     config_dict['alpha'],
        #                     config_dict['her_strategy'],
        #                     config_dict['use_attention'],
        #                     filename=logfile, 
        #                     wb_log=WB_LOG,  
        #                     chkpt_freq=chkpt_freq,
        #                     path=save_path)
    else:
        raise ValueError('Invalid Choice of Algorithm. Exiting ...')

    # Train
    agent.run()