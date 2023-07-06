import numpy as np
import tensorflow as tf

class Encoder:
    """
    A conv net that converts a given image stack into an 1-D feature vector
    args:
        obs_shape: shape of input image stack: (height, width, channels)
        feature_dim: dimension of feature vector
        conv_layers: list of convolutional layers
        dense_layers: list of dense layers
    """
    def __init__(self, obs_shape, feature_dim,
                    conv_layers=[32, 32,],
                    dense_layers=[64,],
                    filter_size=3,
                    strides=1,
                    padding='same',
                    pooling_size=2,
                    learning_rate=1e-3):
        assert len(obs_shape) == 3, 'Input should be an image of shape (h, w, c)' # (height, width, channels)
        self.obs_shape = obs_shape
        self.feature_dim = feature_dim
        self.conv_layers = conv_layers
        self.dense_layers = dense_layers
        self.conv_filter_size = filter_size
        self.strides = strides
        self.padding = padding 
        self.pooling_size = pooling_size
        self.lr = learning_rate 

        # Create the model
        self.model = self._build_model()
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.lr)

    def _build_model(self):
        inputs = tf.keras.layers.Input(shape=self.obs_shape, name="input_layer")
        x = inputs
        for i in range(len(self.conv_layers)):
            x = tf.keras.layers.Conv2D(self.conv_layers[i], 
                        self.conv_filter_size,
                        strides=self.strides, padding=self.padding, activation='relu')(x)
            x = tf.keras.layers.MaxPooling2D(self.pooling_size)(x)
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        x = tf.keras.layers.LayerNormalization()(x)
        for j in range(len(self.dense_layers)):
            x = tf.keras.layers.Dense(self.dense_layers[j], activation='relu')(x)
        x = tf.keras.layers.Dense(self.feature_dim, activation='linear')(x)
        outputs = tf.keras.layers.LayerNormalization(name="output_layer")(x)
        model = tf.keras.models.Model(inputs, outputs, name="encoder")
        model.summary()
        return model

    def __call__(self, state):
        # obs: a tensor or a numpy array
        f = self.model(state)
        return f 

    def compute_loss(self, p, z):
        z = tf.stop_gradient(z)
        p = tf.math.l2_normalize(p, axis=1)
        z = tf.math.l2_normalize(z, axis=1)
        # maximize cosine similarity
        loss = -tf.reduce_mean(tf.reduce_sum((p * z), axis=1))
        return loss 

    def train(self, x_a, x_p):
        # input: numpy arrays or tensors
        with tf.GradientTape() as tape:
            z_a = self.model(x_a)   # anchor
            z_p = self.model(x_p)   # positive
            loss = self.compute_loss(z_a, z_p)
        trainable_params = self.model.trainable_variables
        gradients = tape.gradient(loss, trainable_params)
        self.optimizer.apply_gradients(zip(gradients, trainable_params))
        return loss 


class Decoder:
    """
    A conv net that converts a given feature vector into an image stack
    args:
        obs_shape: (height, width, channels)
        feature_dim: dimension of the feature vector
        conv_layers: list of number of convolutional layers
        dense_layers: list of number of dense layers
        filter_size: size of the convolutional filter

        images are stacked along the depth channel. 
    """
    def __init__(self, obs_shape, feature_dim,
                    filter_size=3,
                    conv_layers=[32, 32,],
                    dense_layers=[64, ],
                    strides=1,
                    padding='same',
                    pooling_size=2,
                    learning_rate=1e-3):
        assert len(obs_shape) == 3, 'Input should be an image stack of shape (h, w, c)' # (height, width, channels)
        self.obs_shape = obs_shape
        self.feature_dim = feature_dim
        self.num_channels = obs_shape[2]    # image shape: (height, width, channels)
        self.conv_layers = conv_layers
        self.dense_layers = dense_layers
        self.conv_filter_size = filter_size
        self.strides = strides
        self.padding = padding
        self.pooling_size = pooling_size
        self.lr = learning_rate 

        # Create the model
        self.model = self._build_model()
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.lr)

    def _build_model(self):
        inputs = tf.keras.layers.Input(shape=self.feature_dim, name="input_layer")
        x = inputs
        for i in range(len(self.dense_layers)):
            x = tf.keras.layers.Dense(self.dense_layers[i], activation='relu')(x)
            x = tf.keras.layers.BatchNormalization()(x)
        
        x = tf.keras.layers.Dense(np.prod(self.obs_shape), activation='relu')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Reshape(self.obs_shape)(x)

        for i in range(len(self.conv_layers)):
            x = tf.keras.layers.Conv2DTranspose(
                                    filters=self.conv_layers[i],
                                    kernel_size=self.filter_size,
                                    strides=self.strides,
                                    padding=self.padding,
                                    activation='relu')(x)
            x = tf.keras.layers.BatchNormalization()(x)
        outputs = tf.keras.layers.Conv2DTranspose(
                                    filters=self.num_channels,
                                    kernel_size=self.filter_size,
                                    padding=self.padding,
                                    strides=self.strides,
                                    activation='sigmoid',
                                    name='decoder_output')(x)
        model = tf.keras.models.Model(inputs, outputs, name="decoder")
        model.summary()
        return model

    def __call__(self, state):
        # obs: a tensor or a numpy array
        f = self.model(state)
        return f



class FeaturePredictor:
    def __init__(self, feature_dim,
                    dense_layers=[256, 128, ],
                    learning_rate=1e-3) -> None:
        self.feature_dim = feature_dim
        self.lr = learning_rate 
        self.dense_layers = dense_layers
        self.model = self._build_model()
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.lr)

    def _build_model(self):
        inputs = tf.keras.layers.Input(shape=(self.feature_dim,), name="input_layer")
        x = inputs
        for i in range(len(self.dense_layers)):
            x = tf.keras.layers.Dense(self.dense_layers[i], activation='relu')(x)
            x = tf.keras.layers.BatchNormalization()(x)
        outputs = tf.keras.layers.Dense(self.feature_dim,
                    activation='linear', name='output_layer')(x)
        model = tf.keras.models.Model(inputs, outputs, name="feature_predictor")
        model.summary()
        return model

    def __call__(self, x):
        return self.model(x)

    def train(self, x, y):
        """
        training minimizes the consistency loss
        x: input feature vector coming from encoder for an augmented image
        y: target feature vector coming from encoder for the original image
        """
        with tf.GradientTape() as tape:
            y_norm = tf.math.l2_normalize(y, axis=1)
            y_pred = self.model(x)
            y_pred_norm = tf.math.l2_normalize(y_pred, axis=1)
            loss = tf.reduce_mean(tf.square(y_norm - y_pred_norm))
        trainable_params = self.model.trainable_variables
        gradients = tape.gradient(loss, trainable_params)
        self.optimizer.apply_gradients(zip(gradients, trainable_params))
        return loss



