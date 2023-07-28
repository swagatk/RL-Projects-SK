import argparse
import collections
import functools
import json
import os
import pathlib
import sys
import time
from typing import Any 


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
EHU = False 
if EHU:
    os.environ['MUJOCO_GL'] = 'egl'


import numpy as np
import tensorflow as tf 
from tensorflow.keras.mixed_precision import experimental as prec 

tf.get_logger().setLevel('ERROR')

from tensorflow_probability import distributions as tfd 

sys.path.append(str(pathlib.Path(__file__).parent))

import models
import tools
import wrappers


def define_config():
    config = tools.AttrDict()

    #General
    config.logdir = pathlib.Path('.')
    config.seed = 0
    config.steps = 5e6
    config.eval_every = 1e4
    config.log_every = 1e3
    config.log_scalars = True
    config.log_images = True
    config.gpu_growth = True
    config.precision = 16
    # Environment.
    config.task = 'dmc_walker_walk'
    config.envs = 1
    config.parallel = 'none'
    config.action_repeat = 2
    config.time_limit = 1000
    config.prefill = 5000
    config.eval_noise = 0.0
    config.clip_rewards = 'none'
    # Model.
    config.deter_size = 200
    config.stoch_size = 30
    config.num_units = 400
    config.dense_act = 'elu'
    config.cnn_act = 'relu'
    config.cnn_depth = 32
    config.pcont = False
    config.free_nats = 3.0
    config.kl_scale = 1.0
    config.pcont_scale = 10.0
    config.weight_decay = 0.0
    config.weight_decay_pattern = r'.*'
    # Training.
    config.batch_size = 50
    config.batch_length = 50
    config.train_every = 1000
    config.train_steps = 100
    config.pretrain = 100
    config.model_lr = 6e-4
    config.value_lr = 8e-5
    config.actor_lr = 8e-5
    config.grad_clip = 100.0
    config.dataset_balance = False
    #config.multi_gpu = False
    # Behavior.
    config.discount = 0.99
    config.disclam = 0.95
    config.horizon = 15
    config.action_dist = 'tanh_normal'
    config.action_init_std = 5.0
    config.expl = 'additive_gaussian'
    config.expl_amount = 0.3
    config.expl_decay = 0.0
    config.expl_min = 0.0
    return config

########################


def count_steps(datadir, config):
  return tools.count_episodes(datadir)[1] * config.action_repeat

def preprocess(obs, config):
  dtype = prec.global_policy().compute_dtype
  obs = obs.copy()
  with tf.device('cpu:0'):
    obs['image'] = tf.cast(obs['image'], dtype) / 255.0 - 0.5
    clip_rewards = dict(none=lambda x: x, tanh=tf.tanh)[config.clip_rewards]
    obs['reward'] = clip_rewards(obs['reward'])
  return obs

def load_dataset(directory, config):
  episode = next(tools.load_episodes(directory, 1))
  types = {k: v.dtype for k, v in episode.items()}
  shapes = {k: (None,) + v.shape[1:] for k, v in episode.items()}
  generator = lambda: tools.load_episodes(
      directory, config.train_steps, config.batch_length,
      config.dataset_balance)
  dataset = tf.data.Dataset.from_generator(generator, types, shapes)
  dataset = dataset.batch(config.batch_size, drop_remainder=True)
  dataset = dataset.map(functools.partial(preprocess, config=config))
  dataset = dataset.prefetch(10)
  return dataset

class Dreamer(tools.Module):
    
    def __init__(self, config, datadir, actspace, writer):
        self._c = config 
        self._actspace = actspace
        self._actdim = actspace.n if hasattr(actspace, 'n') else actspace.shape[0]
        self._writer = writer
        self._random = np.random.RandomState(config.seed)
        with tf.device('cpu:0'):
            self._step = tf.Variable(count_steps(datadir, config), type=tf.int64)

        self._should_pretrain = tools.Once()
        self._should_train = tools.Every(config.train_every)
        self._should_log = tools.Every(config.log_every)
        self._last_log = None 
        self._last_time = time.time()
        self._metrics = collections.defaultdict(tf.metrics.Mean)
        self._metrics['expl_amount']        # create variable for checkpoint
        self._float = prec.global_policy().compute_dtype # changes the data type of layers
        self._strategy = tf.distribute.MirroredStrategy() # Distribute training over multiple GPUs
        with self._strategy.scope():
            self._dataset = iter(self._strategy.experimental_distribute_dataset(
                load_dataset(datadir, self._c)
            ))
            self._build_model()


    def __call__(self, obs, reset, state=None, training=True):
        step = self._step.numpy().item()
        tf.summary.experimental.set_step(step)
        if state is not None and reset.any():       # what is reset?
           mask = tf.cast(1 - reset, self._float)[:, None]
           state = tf.nest.map_structure(lambda x: x * mask, state) # check the shape

        if self._should_train(step):
            log = self._should_log(step)
            n = self._c.pretrain if self._should_pretrain else self._c.train_steps  
            print(f'Training for {n} steps')
            with self._strategy.scope():
                for train_step in range(n):
                    log_images = self._c.log_images and log and train_step == 0
                    self.train(next(self._dataset), log_images)
            
            if log:
              self._write_summaries()
        action, state = self.policy(obs, state, training)
        if training:
           self._step.assign_add(len(reset) * self._c.action_repeat)
        return action, state

    @tf.function
    def policy(self, obs, state, training):
        pass
    
    def _build_model(self):
        acts = dict(
           elu = tf.nn.elu,
           relu = tf.nn.relu,
           swish = tf.nn.swish,
           leaky_relu = tf.nn.leaky_relu
        )
        cnn_act = acts[self._c.cnn_act]
        dense_act = acts[self._c.dense_act]

        self._encode = 

    def _write_summaries(self):
        step = int(self._step.numpy())
        metrics = [(k, float(v.result())) for k, v in self._metrics.items()]
        if self._last_log is not None:
            duration = time.time() - self._last_time
            self._last_time += duration
            metrics.append(('fps', (step - self._last_log) / duration))
        self._last_log = step
        [m.reset_states() for m in self._metrics.values()]
        with (self._c.logdir / 'metrics.jsonl').open('a') as f:
            f.write(json.dumps({'step': step, **dict(metrics)}) + '\n')
        [tf.summary.scalar('agent/' + k, m) for k, m in metrics]
        print(f'[{step}]', ' / '.join(f'{k} {v:.1f}' for k, v in metrics))
        self._writer.flush()


        



