'''
Utilities
'''
import cv2
import matplotlib.pyplot as plt
from matplotlib import animation

# convert images from RGBA to RGB
def rgba2rgb(rgba, background=(255, 255, 255)):
    '''
    RGBA to RGB Conversion
    :param rgba: input RGBA image array of size w x h x 4
    :param background: default value is given
    :return: Returns RGB image of size w x h x 3
    '''
    row, col, ch = rgba.shape

    if ch == 3:
        return rgba

    assert ch == 4, 'RGBA image has 4 channels.'

    rgb = np.zeros((row, col, 3), dtype='float32')
    r, g, b, a = rgba[:, :, 0], rgba[:, :, 1], rgba[:, :, 2], rgba[:, :, 3]

    a = np.asarray(a, dtype='float32') / 255.0

    R, G, B = background

    rgb[:, :, 0] = r * a + (1.0 - a) * R
    rgb[:, :, 1] = g * a + (1.0 - a) * G
    rgb[:, :, 2] = b * a + (1.0 - a) * B


    return np.asarray(rgb, dtype='uint8')


def save_frames_as_gif(frames, path='./', filename='gym_animation.gif'):

    print('Creating GIF Animation File. Wait ...')
    #Mess with this to change frame size
    plt.figure(figsize=(frames[0].shape[1] / 72.0, frames[0].shape[0] / 72.0), dpi=72)

    patch = plt.imshow(frames[0])
    plt.axis('off')

    def animate(i):
        patch.set_data(frames[i])

    anim = animation.FuncAnimation(plt.gcf(), animate, frames = len(frames), interval=50)
    anim.save(path + filename, writer='imagemagick', fps=60)
    print('done!!')


def preprocess_image_input(rgba_img):
    """
    Convert RGBA image to YUV image and return Y channel
    :param rgba_img:
    :return: Y channel of YUV image
    """
    rgb_img = cv2.cvtColor(rgba_img, cv2.COLOR_RGBA2RGB)
    yuv_img = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2YUV)

    return yuv_img[:, :, 0] # return the intensity channel or Y channel
