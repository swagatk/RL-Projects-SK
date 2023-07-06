import tensorflow as tf
import numpy as np
from config import CFG
import tensorflow_probability as tfp

from curl_utils import Config
from feature_extraction import Encoder

### Actor Network
class CurlActor:
    def __init__(self, state_size, action_size,
        action_upper_bound,
        encoder_feature_dim, 
        encoder=None,
        **kwargs) -> None: 
        self.state_size = state_size # shape: (h, w, c)
        self.action_size = action_size
        self.action_uppper_bound = action_upper_bound
        self.encoder_feature_dim = encoder_feature_dim
        self.lr = kwargs.get("lr", 1e-3)
        self.actor_dense_layers = kwargs.get('dense_layers', [128, 64, ]) 
        self.save_model_plot = kwargs.get('save_model_plot', False)
        self.model_name = kwargs.get('model_name', 'actor')
        self.frozen_encoder = kwargs.get('frozen_encoder', False)


        if encoder is None:
            self.encoder = Encoder(obs_shape=self.state_size,
                                feature_dim=self.encoder_feature_dim)
        else:
            self.encoder = encoder
            self.encoder_feature_dim = self.encoder.model.outputs[0].shape[-1]


        self.model = self._build_net()

        if self.frozen_encoder: 
            # Freeze encoder weights during RL training
            self.model.get_layer('encoder').trainable=False 

        self.model.summary()
        self.optimizer = tf.keras.optimizers.Adam(self.lr)

    def _build_net(self):
        inp = tf.keras.layers.Input(shape=self.state_size)
        x = self.encoder(inp)
        for i in range(len(self.actor_dense_layers)):
            x = tf.keras.layers.Dense(self.actor_dense_layers[i],
                        activation='relu')(x)
        mu = tf.keras.layers.Dense(self.action_size[0],
                            activation='tanh')(x)
        log_sig = tf.keras.layers.Dense(self.action_size[0])(x)
        mu = mu * self.action_uppper_bound
        model = tf.keras.Model(inputs=inp, outputs=[mu, log_sig],
                            name=self.model_name)
        if self.save_model_plot:
            tf.keras.utils.plot_model(model,
                                to_file='actor_network.png',
                                show_shapes=True,
                                show_layer_names=True)
        return model

    def  __call__(self, state):
        mu, log_sig = self.model(state)
        std = tf.math.exp(log_sig)
        return mu, std 

    def policy(self, state):

        #mean, std = self.__call__(state)
        mean, std = self(state)

        # sample actions from normal distribution
        pi = tfp.distributions.Normal(mean, std)
        action_ = pi.sample()
        log_pi_ = pi.log_prob(action_)
        
        # Apply squashing function
        action = tf.clip_by_value(action_, 
                    -self.action_uppper_bound,
                    self.action_uppper_bound)
        
        log_pi_a = log_pi_ - tf.reduce_sum(
            tf.math.log(tf.keras.activations.relu(1 - action ** 2) + 1e-6),
                            axis=-1, keepdims=True)
        return action, log_pi_a 

    def train(self):
        pass

    def save_weights(self, filename):
        self.model.save_weights(filename, save_format='h5')

    def load_weights(self, filename):
        self.model.load_weights(filename)


if __name__ == '__main__':
    config = Config.from_json(CFG)
    actor = CurlActor(state_size=(100, 100, 9),
                        action_size=(3, 1),
                        action_upper_bound=[1, 1, 1],
                        encoder_feature_dim=50,
                        **config.actor.__dict__)

    print('Type of config:', config.actor.__dict__)