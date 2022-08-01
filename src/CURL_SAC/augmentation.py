
import numpy as np
from skimage.util import view_as_windows

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

def center_crop_image(imgs, out_h, out_w=None):
    '''
    args: 
        image: input image of shape (h, w, c) or (b, h, w, c)
        out_h: output height
        out_w: output width

    returns:
        cropped image of shape (-1, h, w, c)
    '''
    img_array = np.asarray(imgs)
    
    if len(img_array.shape) == 3:   # (h, w, c)
        img_h = img_array.shape[0]
        img_w = img_array.shape[1]
    elif len(img_array.shape) == 4:     # (B, H, W, C)
        img_h = img_array.shape[1]
        img_w = img_array.shape[2]

    if out_w is None:
        out_w = out_h
        
    top = (img_h - out_h) // 2
    left = (img_w - out_w) // 2
    cropped_images = img_array[..., top:top+out_h, left:left+out_w, :]
    return cropped_images


if __name__ == '__main__':
    img = np.random.rand(3, 100, 100, 3)
    img2 = np.random.rand(100, 100, 3)
    img_rc = random_crop(img, 50, 50)
    print(img_rc.shape)
    img_cc = center_crop_image(img, 50, 50)
    print(img_cc.shape)
    img2_cc = center_crop_image(img2, 50, 50)
    print(img2_cc.shape)