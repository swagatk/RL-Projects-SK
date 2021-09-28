import gym
import numpy as np
import tensorflow as tf
#import imageio
import matplotlib.pyplot as plt
from pybullet_envs.bullet.racecarZEDGymEnv import RacecarZEDGymEnv
from CustomGymWrapper import ObsvnResizeTimeLimitWrapper
from VariationAutoEncoder import VAE

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

# observation is an image
env = ObsvnResizeTimeLimitWrapper(
                    RacecarZEDGymEnv(isDiscrete=False, renders=False),
                    shape=40,
                    max_steps=20
                    )


print('shape of Observation space: ', env.observation_space.shape)
print('shape of Action space: ', env.action_space.shape)
print('Reward Range: ', env.reward_range)
print('Action High value: ', env.action_space.high)
print('Action Low Value: ', env.action_space.low)

print('Data Collection from the Gym Environment:')
data_buffer = [] 
time_steps = 0
for ep in range(2000):
    obsv = np.asarray(env.reset(), dtype=np.float32) / 255.0

    while True:
        data_buffer.append(np.expand_dims(obsv, axis=0))
        action = env.action_space.sample()
        next_obsv, reward, done, _ = env.step(action)

        next_obsv = np.asarray(next_obsv, dtype=np.float32) / 255.0
        obsv = next_obsv

        time_steps += 1
        if done:
            break
env.close()

dataset = np.concatenate(data_buffer, axis=0)
print('dataset shape:', np.shape(dataset))

print('Training the model ... wait!')
input_shape = env.observation_space.shape
latent_dim = 10 
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
vae.save_model('./trained_models/')



