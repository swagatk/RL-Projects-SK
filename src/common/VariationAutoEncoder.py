"""
Variational Auto Encoder

22/01/2022: 
    - It now supports stacked frames
"""
import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, Input, Flatten, Dense, Lambda, Reshape
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.datasets import mnist
from tensorflow.keras.losses import binary_crossentropy
from tensorflow.keras import backend as K
import matplotlib.pyplot as plt
from tensorflow.python.eager.context import num_gpus


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


# create a sampling layer
class Sampling(tf.keras.layers.Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""

    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon


####################
## Encoder
##################
class Encoder(keras.Model):
    def __init__(self, input_shape, latent_dim, **kwargs) -> None:
        super(Encoder, self).__init__(**kwargs)
        self.inp_shape = input_shape
        self.latent_dim = latent_dim
        self.optimizer = tf.keras.optimizers.Adam(1e-3)
        self.model = self._build_model()

    def _build_model(self):
        i = tf.keras.Input(shape=self.inp_shape, name='encoder_input')
        cx = Conv2D(filters=32, kernel_size=3, strides=2, padding='same', activation='relu')(i)
        cx = BatchNormalization()(cx)
        cx = Conv2D(filters=64, kernel_size=3, strides=2, padding='same', activation='relu',
                                                    name='last_conv')(cx)
        cx = BatchNormalization()(cx)
        x = Flatten()(cx)
        x = Dense(20, activation='relu')(x)
        x = BatchNormalization()(x)
        z_mean = Dense(self.latent_dim, name='latent_mu')(x)
        z_log_var = Dense(self.latent_dim, name='latent_sigma')(x)
        z = Sampling()([z_mean, z_log_var])
        encoder = Model(i, [z_mean, z_log_var, z], name='encoder')
        encoder.summary()
        return encoder

    def get_last_conv_shape(self):
        # Get shape of last conv layer in the encoder
        last_conv_layer = self.model.get_layer('last_conv')
        conv_shape = K.int_shape(last_conv_layer.output)
        return conv_shape

    def __call__(self, input_obs, deterministic=True):
        # input is a tensor
        # output is a tuple

        z_mean, z_log_var, z = self.model(input_obs)
        if deterministic:
            return z_mean
        else:
            return z_mean, z_log_var, z 

    def viz_latent_space(self, data):
        input_data, target_data = data
        mu, _, _ = self.model.predict(input_data)
        plt.figure(figsize=(8, 10))
        plt.scatter(mu[:, 0], mu[:, 1], c=target_data)
        plt.xlabel('z - dim 1')
        plt.ylabel('z - dim 2')
        plt.colorbar()
        plt.show()

    def save_model(self, filename):
        self.model.save_weights(filename)

    def load_model(self, filename):
        self.model.load_weights(filename)
        print('Encoder weights are loaded.')



#####################3
## Decoder
#####################333

class Decoder(keras.Model):
    def __init__(self, out_shape, conv_shape, latent_dim, **kwargs) -> None:
        super(Decoder, self).__init__(**kwargs)
        self.out_shape = out_shape
        self.conv_shape = conv_shape
        self.latent_dim = latent_dim
        self.num_channels = self.out_shape[2]
        self.optimizer = tf.keras.optimizers.Adam()
        self.model = self._build_model()

    def _build_model(self):
        # Definition
        d_i = Input(shape=(self.latent_dim, ), name='decoder_input')
        x = Dense(self.conv_shape[1] * self.conv_shape[2] * self.conv_shape[3], activation='relu')(d_i)
        x = BatchNormalization()(x)
        x = Reshape((self.conv_shape[1], self.conv_shape[2], self.conv_shape[3]))(x)
        cx = Conv2DTranspose(filters=64, kernel_size=3, strides=2, padding='same', activation='relu')(x)
        cx = BatchNormalization()(cx)
        cx = Conv2DTranspose(filters=32, kernel_size=3, strides=2, padding='same',  activation='relu')(cx)
        cx = BatchNormalization()(cx)
        o = Conv2DTranspose(filters=self.num_channels, kernel_size=3, activation='sigmoid', padding='same', name='decoder_output')(cx)
        decoder = Model(d_i, o, name='decoder')
        decoder.summary()
        return decoder

    def __call__(self, inp_data):
        # input is a tensor
        decoder_output = self.model(inp_data)
        return decoder_output

    def save_model(self, filename):
        self.model.save_weights(filename)

    def load_model(self, filename):
        self.model.load_weights(filename)
        print('Decoder weights are loaded.')
        


###########################333
## VAE 
#############################3
class VAE(keras.Model):
    def __init__(self, obs_shape, latent_dim, **kwargs):
        super(VAE, self).__init__(**kwargs)
        self.input_obs_shape = obs_shape
        self.latent_dim = latent_dim
        self.encoder = Encoder(self.input_obs_shape, self.latent_dim)
        self.conv_shape = self.encoder.get_last_conv_shape()
        self.decoder = Decoder(self.input_obs_shape, self.conv_shape, self.latent_dim)
        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = keras.metrics.Mean(
            name="reconstruction_loss"
        )
        self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
        ]

    def train_step(self, data):
        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder(data, deterministic=False)
            reconstruction = self.decoder(z)
            reconstruction_loss = tf.reduce_mean(
                tf.reduce_sum(
                    keras.losses.binary_crossentropy(data, reconstruction), axis=(1, 2) # check dimension
                )
            )
            kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
            kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
            total_loss = reconstruction_loss + kl_loss
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }

    def test_step(self, data):
        z_mean, z_log_var, z = self.encoder(data, deterministic=False)
        reconstruction = self.decoder(z)
        reconstruction_loss = tf.reduce_mean(
            tf.reduce_sum(
                keras.losses.binary_crossentropy(data, reconstruction), axis=(1, 2) # check dimension
            )
        )
        kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
        kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
        total_loss = reconstruction_loss + kl_loss

        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)

        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }

    def viz_decoded(self, inp_data):
        assert inp_data.ndim == 4, 'Image should have 4 dimension: (-1, h, w, c)'


        z_mean = self.encoder(inp_data, deterministic=True)
        out_data = self.decoder(z_mean)


        if inp_data.shape[3] > 3:
            inp_data = inp_data[:, :, :, :3] # retain only first 3 channels
        
        if out_data.shape[3] > 3:
            out_data = out_data[:, :, :, :3] # retain only first 3 channels

        rows = len(inp_data)
        plt.figure()
        fig, axes = plt.subplots(rows,2)
        for i in range(rows):
            axes[i][0].imshow(inp_data[i]) 
            axes[i][1].imshow(out_data[i])
            axes[i][0].axis('off')
            axes[i][1].axis('off')
            if i == 0:
                axes[i][0].set_title('Original Image')
                axes[i][1].set_title('Reconstructed Image')
        plt.tight_layout()
        plt.savefig('./reconst.png')

    def save_model(self, save_path):
        os.makedirs(save_path, exist_ok=True)
        enc_file = save_path + 'enc_wts.h5'
        dec_file = save_path + 'dec_wts.h5'
        self.encoder.save_model(enc_file)
        self.decoder.save_model(dec_file)

    def load_model(self, load_path):
        enc_file = load_path + 'enc_wts.h5'
        dec_file = load_path + 'dec_wts.h5'
        self.encoder.load_model(enc_file)
        self.decoder.load_model(dec_file)



if __name__ == '__main__':
    (x_train, _), (x_test, _) = keras.datasets.mnist.load_data()
    mnist_digits = np.concatenate([x_train, x_test], axis=0)
    mnist_digits = np.expand_dims(mnist_digits, -1).astype("float32") / 255

    print('MNIST digit shape:', np.shape(mnist_digits))

    inp_shape = np.shape(mnist_digits)[1:]
    latent_dim = 2

    vae = VAE(obs_shape=inp_shape, latent_dim=latent_dim)
    vae.compile(optimizer=keras.optimizers.Adam())
    vae.fit(mnist_digits, epochs=30, batch_size=128)    
    vae.viz_decoded(mnist_digits[0:4])
    




