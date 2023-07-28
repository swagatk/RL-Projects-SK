import datetime
import io
import pathlib
import pickle
import re
from typing import Any
import uuid

import gym
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.keras.mixed_precision import experimental as prec
from tensorflow_probability import distributions as tfd


class AttrDict(dict):

  __setattr__ = dict.__setitem__
  __getattr__ = dict.__getitem__


class Module(tf.Module):

  def save(self, filename):
    # convert tensors to numpy array before storing
    values = tf.nest.map_structure(lambda x: x.numpy(), self.variables)
    with pathlib.Path(filename).open('wb') as f:
      pickle.dump(values, f)

  def load(self, filename):
    with pathlib.Path(filename).open('rb') as f:
      values = pickle.load(f)
    tf.nest.map_structure(lambda x, y: x.assign(y), self.variables, values)

  def get(self, name, ctor, *args, **kwargs):
    # Create or get layer by name to avoid mentioning it in the constructor.
    if not hasattr(self, '_modules'):
      self._modules = {}
    if name not in self._modules:
      self._modules[name] = ctor(*args, **kwargs) # constructor
    return self._modules[name]
    

class Every:

  def __init__(self, every):
    self._every = every
    self._last = None

  def __call__(self, step):
    if self._last is None:
      self._last = step
      return True
    if step >= self._last + self._every:
      self._last += self._every
      return True
    return False


class Once:

  def __init__(self):
    self._once = True

  def __call__(self):
    if self._once:
      self._once = False
      return True
    return False


class TanhBijector(tfp.bijectors.Bijector):
  
  def __init__(self, validate_args=False, name='tanh'):
    super().__init__(
      forward_min_event_ndims=0,
      validate_args=validate_args,
      name=name
    )
    
  def _forward(self, x):
    return tf.nn.tanh(x) 
  
  def _inverse(self, y):
    dtype = y.dtype
    y = tf.cast(y, tf.float32)
    y = tf.where(
      tf.less_equal(tf.abs(y), 1.),
      tf.clip_by_value(y, -0.99999997, 0.99999997), y)
    y = tf.atanh(y)
    y = tf.cast(y, dtype)
    return y 
  
  def _forward_log_det_jacobian(self, x):
    log2 = tf.math.log(tf.constant(2.0, dtype=x.dtype))
    return 2.0 * (log2 - x - tf.nn.softplus(-2.0 * x))


class SampleDist:
  def __init__(self, dist, samples=100) -> None: # dist is a tfd distribution
    self._dist = dist 
    self._samples = samples

  @property
  def name(self):
    return 'SampleDist'
  
  def __getattr__(self, name):
    return getattr(self._dist, name)    # not sure how it works
  

  def mean(self):
    samples = self._dist.sample(self._samples)
    return tf.reduce_mean(samples, 0)
  
  def entropy(self):
    sample = self._dist.sample(self._samples)
    logprob = self.log_prob(sample)     # NOT DEFINED
    return -tf.reduce_mean(logprob, 0)
  

class OneHotDist:

  def __init__(self, logits=None, probs=None) -> None:
    self._dist = tfd.Categorical(logits=logits, probs=probs)
    self._num_classes = self.mean().shape[-1]
    self._dtype = prec.global_policy().compute_dtype 

  @property
  def name(self):
    return 'OneHotDist'
  
  def __getattr__(self, name):
    return getattr(self._dist, name)
  
  def prob(self, events):
    indices = tf.argmax(events, axis=-1)
    return self._dist.prob(indices) 
  
  def log_prob(self, events):
    indices = tf.argmax(events, axis=-1)
    return self._dist.log_prob(indices)
  
  def mean(self):
    return self._dist.probs_parameter() 
  
  def mode(self):
    return self._one_hot(self._dist.mode())
  
  def sample(self, amount=None):
    amount = [amount] if amount else []
    indices = self._dist.sample(*amount)
    sample = self._one_hot(indices)
    probs = self._dist.probs_parameter()
    sample += tf.cast(probs - tf.stop_gradient(probs), self._dtype)
    return sample 
  
  def _one_hot(self, indices):
    return tf.one_hot(indices, self._num_classes, dtype=self._dtype)



############
def count_episodes(directory):
  filenames = directory.glob('*.npz')
  lengths = [int(n.rsplit('-', 1)[-1]) - 1 for n in filenames]
  episodes, steps = len(lengths), sum(lengths)
  return episodes, steps


def load_episodes(directory, rescan, length=None, balance=False, seed=0):
  directory = pathlib.Path(directory).expanduser()
  random = np.random.RandomState(seed)
  cache = {}
  while True:
    for filename in directory.glob('*.npz'):
      if filename not in cache:
        try:
          with filename.open('rb') as f:
            episode = np.load(f)
            episode = {k: episode[k] for k in episode.keys()}
        except Exception as e:
          print(f'Could not load episode: {e}')
          continue
        cache[filename] = episode
    keys = list(cache.keys())
    for index in random.choice(len(keys), rescan):
      episode = cache[keys[index]]
      if length:
        total = len(next(iter(episode.values())))
        available = total - length
        if available < 1:
          print(f'Skipped short episode of length {available}.')
          continue
        if balance:
          index = min(random.randint(0, total), available)
        else:
          index = int(random.randint(0, available))
        episode = {k: v[index: index + length] for k, v in episode.items()}
      yield episode


# what does it do? 
def static_scan(fn, inputs, start, reverse=False):
  last = start
  outputs = [[] for _ in tf.nest.flatten(start)]
  indices = range(len(tf.nest.flatten(inputs)[0]))
  if reverse:
    indices = reversed(indices)
  for index in indices:
    inp = tf.nest.map_structure(lambda x: x[index], inputs)
    last = fn(last, inp)
    [o.append(l) for o, l in zip(outputs, tf.nest.flatten(last))]
  if reverse:
    outputs = [list(reversed(x)) for x in outputs]
  outputs = [tf.stack(x, 0) for x in outputs]
  return tf.nest.pack_sequence_as(start, outputs)
  
