import imageio
import matplotlib.pyplot as plt
import numpy as np
import pybullet_envs as pe
import datetime
from numpngw import write_apng
from IPython.display import Image, display


from stable_baselines import PPO2
print('Start learning, wait a minute')
start = datetime.datetime.now()
model = PPO2("MlpPolicy", 'CartPoleContinuousBulletEnv-v0').learn(40000)
end = datetime.datetime.now()
print('Training time:', end-start)
print('done learning')

images = []
obs = model.env.reset()
img = model.env.render(mode='rgb_array')
for i in range(200):
    action, _ = model.predict(obs)
    obs, reward, done, _ = model.env.step(action)
    img = model.env.render(mode='rgb_array')
    images.append(img)
model.env.close()
write_apng('anim.png', images, delay=20)
Image(filename='anim.png')


