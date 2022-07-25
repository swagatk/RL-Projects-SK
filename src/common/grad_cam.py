'''
Routines for creating grad-CAM heatmaps
'''

import tensorflow as tf
from tensorflow import keras
import matplotlib.cm as cm
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


## create grad-CAM heatmap
def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    # First, we create a model that maps the input image to the activations
    # of the last conv layer as well as the output predictions
    img_array = tf.convert_to_tensor(tf.expand_dims(img_array, axis=0))
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
    )

    # Then, we compute the gradient of the top predicted class for our input image
    # with respect to the activations of the last conv layer
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    # This is the gradient of the output neuron (top predicted or chosen)
    # with regard to the output feature map of the last conv layer
    grads = tape.gradient(class_channel, last_conv_layer_output)

    # print('shape of grads:', tf.shape(grads))


    # This is a vector where each entry is the mean intensity of the gradient
    # over a specific feature map channel
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # We multiply each channel in the feature map array
    # by "how important this channel is" with regard to the top predicted class
    # then sum all the channels to obtain the heatmap class activation
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # For visualization purpose, we will also normalize the heatmap between 0 & 1
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()


def save_and_display_gradcam(img_array, heatmap, cam_path="cam.jpg", alpha=0.4):
    # Load the original image
    # img = keras.preprocessing.image.load_img(img_path)
    # img = keras.preprocessing.image.img_to_array(img)

    img = np.uint(255 * img_array)
    #print('shape of input image:', np.shape(img))

    # Rescale heatmap to a range 0-255
    heatmap = np.uint8(255 * heatmap)

    # Use jet colormap to colorize heatmap
    jet = cm.get_cmap("jet")

    # Use RGB values of the colormap
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap]

    # Create an image with RGB colorized heatmap
    jet_heatmap = keras.preprocessing.image.array_to_img(jet_heatmap)
    jet_heatmap = jet_heatmap.resize((img.shape[1], img.shape[0]))
    jet_heatmap = keras.preprocessing.image.img_to_array(jet_heatmap)

    # Superimpose the heatmap on original image
    superimposed_img = jet_heatmap * alpha + img

    superimposed_img = keras.preprocessing.image.array_to_img(superimposed_img)

    # save the superimposed image
    keras.preprocessing.image.save_img(cam_path, superimposed_img)


    # plt.figure(4)
    # plt.imshow(superimposed_img)


    # Display Grad CAM
    #display(Image(cam_path))
    # image = Image.open(cam_path)
    # image.show()
    return superimposed_img


def grad_cam2(img_array, model1, model2, conv_layer_name1, conv_layer_name2):
    # I don't know what am I doing here. Its incomplete
    # make sure that the input image is in same format as defined in the model definition
    img_array = tf.convert_to_tensor(tf.expand_dims(img_array, axis=0))
    grad_model_1 = tf.keras.models.Model(
        [model1.inputs], [model1.get_layer(conv_layer_name1).output, model1.output]
    )
    grad_model_2 = tf.keras.models.Model(
        [model2.inputs], [model2.get_layer(conv_layer_name2).output, model2.output]
    )

    # Then, we compute the gradient of the top predicted class for our input image
    # with respect to the activations of the last conv layer
    with tf.GradientTape() as tape_1:
        conv_layer_output_1, preds1 = grad_model_1(img_array)
        pred_index = tf.argmax(preds1[0])
        class_channel = preds1[:, pred_index]
        print('shape of preds1:', tf.shape(preds1))
    grads_1 = tape_1.gradient(class_channel, conv_layer_output_1)
    print('shape of grads_1:', tf.shape(grads_1))

    with tf.GradientTape() as tape_2:
        conv_layer_output_2, preds2 = grad_model_2(img_array)
        print('shape of preds2:', tf.shape(preds2))
    grads_2 = tape_2.gradient(preds2[0], conv_layer_output_2)
    print('shape of grads_2:', tf.shape(grads_2))


