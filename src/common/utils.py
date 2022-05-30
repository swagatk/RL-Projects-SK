'''
Utility functions

13/01/2022: Added the following functions:
            - prepare_stacked_images()
            - visualize_stacked_images()
'''
import os
import signal
import time
import numpy as np
import matplotlib.pyplot as plt
from skimage.util import view_as_windows
import tensorflow as tf

#################################
def uniquify(path):
    # creates a unique file name by adding an incremented number
    # to the existing filename
    filename, extension = os.path.splitext(path)
    counter = 1
    while os.path.exists(path):
        path = filename + '_' + str(counter) + extension
        counter += 1
    return path

########################
# graceful exit
##################
class GracefulExiter():
    def __init__(self):
        self.state = False
        signal.signal(signal.SIGINT, self.change_state)

    def change_state(self, signum, frame):
        print('exit flag set to True (repeat to exit now)')
        signal.signal(signal.SIGINT, signal.SIG_DFL)
        self.state = True

    def exit(self):
        return self.state

########################
## Create stacked frames
########################

def prepare_stacked_images(img_buffer, stack_size=2):
    # input : list of images of shape: (h, w, c)
    # output: stacked frames of shape: (h, w, c*stack_size)
    if stack_size > 1:
        temp_list = []
        for i in range(stack_size):
            if i < len(img_buffer):
                temp_list.append(img_buffer[-1-i])      # fill in the reversed order
            else:
                temp_list.append(img_buffer[-1])        # last element

        stacked_img = np.dstack(temp_list)      # stack the images along depth channel
        return stacked_img      # check the shape:  (h, w, c*stack_size)
    else:
        return img_buffer[-1]   # return the last image in the buffer

def visualize_stacked_images(stacked_img, save_fig=False, fig_name='stacked_img.png'):
    # input : stacked frames of shape: (h, w, c*stack_size)
    # output: Plot of stacked images
    assert(len(stacked_img.shape) == 3), "stacked_img must have 3 dimensions"
    assert(stacked_img.shape[2] % 3 == 0), "stacked_img must have 3x channels"
    image_list = np.dsplit(stacked_img, int(stacked_img.shape[2]//3))      # split the stacked image into sections of 3 channels

    rows = int(np.ceil(len(image_list) / 2)) # upper bound for the number of rows
    cols = int(np.ceil(len(image_list) / 2)) # upper bound for the number of columns
    fig, axs = plt.subplots(rows, cols, figsize=(10, 10))
    fig.suptitle('Stacked Images', fontsize=12)
    for i in range(rows):
        for j in range(cols):
            k = i*cols+j
            if k < len(image_list):
                axs[i, j].imshow(image_list[k])
                axs[i, j].axis('off')
                axs[i, j].set_title('Frame {}'.format(k))
    fig.tight_layout()
    plt.show()
    if save_fig:
        plt.savefig(uniquify(fig_name))

###################################33
# Random crop 
# Source: https://github.com/MishaLaskin/curl/blob/master/utils.py
###################################33
def random_crop(imgs, out_h, out_w=None):
    """
    args:
        imgs: batch images with size (B, H, W, C)
        out_h: output height
        out_w: output width

    returns:
        cropped images with size (B, H, W, C)
    """
    img_array = np.asarray(imgs)
    n = img_array.shape[0]      # batch size
    img_h = img_array.shape[1]  # height
    img_w = img_array.shape[2]  # width

    if out_w is None:
        out_w = out_h

    assert img_h > out_h and img_w > out_w, "Image size must be greater than output size"


    crop_max_h = img_h - out_h 
    crop_max_w = img_w - out_w

    # create 
    h1_idx = np.random.randint(0, crop_max_h, n)
    w1_idx = np.random.randint(0, crop_max_w, n)

    # create all sliding windows combination of size: output_size
    windows =  view_as_windows(
        img_array, (1, out_h, out_w, 1)
    )[..., 0, :, :, 0]
    cropped_imgs = windows[np.arange(n), h1_idx, w1_idx]
    cropped_imgs = cropped_imgs.transpose(0, 2, 3, 1)
    return cropped_imgs

def center_crop_image(image, out_h, out_w=None):
    '''
    args: 
        image: input image of shape (h, w, c)
        output_h: output height
        output_w: output width

    returns:
        cropped image of shape (h, w, c)
    '''
    if out_w is None:
        out_w = out_h
        
    h, w = image.shape[:2]  # height and width of the image
    top = (h - out_h) // 2
    left = (w - out_w) // 2
    cropped_image = image[top:top+out_h, left:left+out_w, :]
    return cropped_image


def set_seed_everywhere(seed=42, env=None) -> None:
    tf.random.set_seed = seed 
    np.random.seed(seed)
    if env is not None:
        env.seed(seed)
    

