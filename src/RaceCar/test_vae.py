import numpy as np
import tensorflow as tf
from VariationAutoEncoder import Encoder

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
###########################33

input_shape = (40, 40, 3)
latent_dim = 2
encoder = Encoder(input_shape, latent_dim)
encoder.model.load_weights('./trained_models/enc_wts.h5')
