'''
We will use a pre-trained VGG model to extract features from the input image
Tensorflow 2.0 code
'''
import PIL.Image as Image
import matplotlib as plt
import tensorflow as tf
from keras_preprocessing import image
from keras_applications.vgg16 import VGG16
from keras_applications.vgg16 import preprocess_input


class FeatureExtractor:
    def __init__(self, network_arch='vgg'):
        self.network_arch = network_arch

        if network_arch == 'vgg':
            self.model = VGG16(weights='imagenet', include_top=False)
        else:
            raise ValueError('Not defined for other values. Use only vgg')
        self.model.summary()

    def extract_features(self, input_image):
        '''
        :param input_image: RGB image array of size w x h x 3
        :return: 1-D feature vector of size: 7x7x512 = 25,088
        '''
        img_data = preprocess_input(input_image)
        features = self.model.predict(img_data)
        features = features.flatten()
        return features


