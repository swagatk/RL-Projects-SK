import numpy as np
import tensorflow as tf
#import imageio
import matplotlib.pyplot as plt
import os
import sys
current_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(current_dir)
sys.path.append(os.path.dirname(current_dir))

# local imports
import pybullet_multigoal_gym as pmg
from common.VariationAutoEncoder import VAE

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
if device_name != '/device:GPU:0':
    raise SystemError('GPU device not found')
print('Found GPU at: {}'.format(device_name))
##############################################


def vae_train(env, ep_max=20000, latent_dim=10):

    print('shape of Observation space: ', env.observation_space['observation'].__getattribute__('shape'))
    print('shape of Action space: ', env.action_space.shape)
    print('Reward Range: ', env.reward_range)
    print('Action High value: ', env.action_space.high)
    print('Action Low Value: ', env.action_space.low)

    print('Data Collection from the Gym Environment:')
    data_buffer = [] 
    time_steps = 0
    for _ in range(ep_max):
        obsv = env.reset()
        state = np.asarray(obsv['observation'], dtype=np.float32) / 255.0

        while True:
            data_buffer.append(np.expand_dims(state, axis=0))
            action = env.action_space.sample()
            next_obsv, _, done, _ = env.step(action)

            next_state = np.asarray(next_obsv['observation'], dtype=np.float32) / 255.0
            state = next_state

            time_steps += 1
            if done:
                break
    env.close()

    dataset = np.concatenate(data_buffer, axis=0)
    print('dataset shape:', np.shape(dataset))

    print('Training the VAE model ... wait!')
    input_shape = env.observation_space['observation'].__getattribute__('shape')
    print('Input shape:', input_shape)
    vae = VAE(input_shape, latent_dim)
    vae.compile(optimizer=tf.keras.optimizers.Adam())
    history = vae.fit(dataset,  validation_split=0.33, epochs=1000, batch_size=200)
    print('Training completed!')

    # plotting
    print(history.history.keys())
    plt.figure(1)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('loss')
    plt.xlabel('Epochs')
    plt.legend(['loss', 'val_loss'], loc='best')
    plt.savefig('./loss.png')
    #plt.show()

    # reconstructed images
    indices = np.random.choice(range(len(dataset)), 6)
    vae.viz_decoded(dataset[indices])

    # Save model
    vae.save_model('./vae_models/')



