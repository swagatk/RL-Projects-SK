import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as TF
import torchvision.datasets as datasets
import utils 
from skimage.util.shape import view_as_windows
import random
import os
from PIL import ImageFile
# use this if you are getting unidentifiedImageError
ImageFile.LOAD_TRUNCATED_IMAGES=True

def random_crop(imgs, output_size):
     """
        Vectorized way to do random crop using sliding windows
        and picking out random ones
        args:
        imgs, batch images with shape (B,C,H,W)
    """
     n = imgs.shape[0]  # batch_size
     img_size = imgs.shape[-1]
     crop_max = img_size - output_size
     imgs = np.transpose(imgs, (0, 2, 3, 1)) # convert to channel last image
     w1 = np.random.randint(0, crop_max, n)
     h1 = np.random.randint(0, crop_max, n)
     # creates all sliding windows combinations of size (output_size)
     windows = view_as_windows(imgs, (1, output_size, output_size, 1))[..., 0, :, :, 0]
     return windows[np.arange(n), w1, h1]

def center_crop_image(image, output_size):
    """
    Input Image size: C, H, W
    """
    # crop single image
    h, w = image.shape[-2], image.shape[-1]
    #h, w = 100, 100
    new_h, new_w = output_size, output_size

    top = (h - new_h) // 2
    left = (w - new_w) // 2

    image = image[:, top : top + new_h, left : left + new_w]
    return image

def center_crop_image_batch(image, output_size):
    """
    input image size: N, C, H, W
    Assuming H=W (square image)
    """
    h, w = image.shape[2], image.shape[3]
    new_h, new_w = output_size, output_size

    assert new_h < h and new_w < w, "output image size must be smaller than the original size"

    top = (h - new_h) // 2
    left = (w - new_w) // 2

    image = image[:, :, top : top + new_h, left : left + new_w]
    return image


def center_crop_image_batch_nw(imgs, out_h, out_w=None):
    '''
    args: 
        image: input image of shape (c, h, w) or (b, c, h, w)
        out_h: output height
        out_w: output width
    returns:
        cropped image of shape (-1, h, w, c)
    '''
    img_array = np.asarray(imgs)
    
    if len(img_array.shape) == 3:   # (c, h, w)
        img_h = img_array.shape[1]
        img_w = img_array.shape[2]
    elif len(img_array.shape) == 4:     # (B, C, H, W)
        img_h = img_array.shape[2]
        img_w = img_array.shape[3]

    if out_w is None:
        out_w = out_h
        
    top = (img_h - out_h) // 2
    left = (img_w - out_w) // 2
    cropped_images = img_array[..., top:top+out_h, left:left+out_w, :]
    return cropped_images


def random_conv(imgs, out_size):
    """Applies a random conv2d, deviates slightly from https://arxiv.org/abs/1910.05396
    input: images of shape (n, c, h, w)
    """
    
    imgs = torch.from_numpy(imgs).float()
    n, c, h, w = imgs.shape
    #print(h, w)
    for i in range(n):
        weights = torch.randn(3, 3, 3, 3).to(imgs.device)
        temp_imgs = imgs[i : i + 1].reshape(-1, 3, h, w) / 255.0
        temp_imgs = F.pad(temp_imgs, pad=[1] * 4, mode="replicate")
        out = torch.sigmoid(F.conv2d(temp_imgs, weights)) * 255.0
        total_out = out if i == 0 else torch.cat([total_out, out], axis=0)
    return np.array(total_out.reshape(n, c, h, w))


def random_convolution_4d(imgs, kernel_size=3, channels_out=None):
    """
    performs random convolution to input tensors
    input: imgs of shape (n, c, h, w)
    output: imgs of shape (n, c, h, w)
    """
    channels_in = imgs.shape[1]
    if channels_out is None:
        channels_out = imgs.shape[1]
    weight = torch.randn(channels_out, channels_in, kernel_size, kernel_size)
    weight = weight / torch.max(weight)
    output_tensor = torch.nn.functional.conv2d(
        imgs, weight, padding='same'
    )
    return output_tensor
places_dataloader = None
places_iter = None


def _load_places(batch_size=128, image_size=84, num_workers=1, use_val=False):
	global places_dataloader, places_iter
	partition = 'val' if use_val else 'train'
	print(f'Loading {partition} partition of places365_standard...')
	for data_dir in utils.load_config('datasets'):
		if os.path.exists(data_dir):
			fp = os.path.join(data_dir, 'places365_standard', partition)
			if not os.path.exists(fp):
				print(f'Warning: path {fp} does not exist, falling back to {data_dir}')
				fp = data_dir
			places_dataloader = torch.utils.data.DataLoader(
				datasets.ImageFolder(fp, TF.Compose([
					TF.RandomResizedCrop(image_size),
					TF.RandomHorizontalFlip(),
					TF.ToTensor()
				])),
				batch_size=batch_size, shuffle=True,
				num_workers=num_workers, pin_memory=True, persistent_workers=True)
			places_iter = iter(places_dataloader)
			break
	if places_iter is None:
		raise FileNotFoundError('failed to find places365 data at any of the specified paths')
	print('Loaded dataset from', data_dir)


def _get_places_batch(batch_size):
	global places_iter
	try:
		imgs, _ = next(places_iter)
		if imgs.size(0) < batch_size:
			places_iter = iter(places_dataloader)
			imgs, _ = next(places_iter)
	except StopIteration:
		places_iter = iter(places_dataloader)
		imgs, _ = next(places_iter)
	return imgs.cuda()


def random_overlay(x, dataset='places365_standard'):
    """Randomly overlay an image from Places"""
    global places_iter
    alpha = 0.5
    #print("inside random overlay", x.shape, x.shape[0])

    if dataset == 'places365_standard':
        if places_dataloader is None:
            _load_places(batch_size=x.shape[0], image_size=x.shape[-1])
        imgs = _get_places_batch(batch_size=x.shape[0]).repeat(1, x.shape[1]//3, 1, 1)
    else:
        raise NotImplementedError(f'overlay has not been implemented for dataset "{dataset}"')
    #print("type of operands", type(x), type(imgs))

    return ((1-alpha)*(x/255.) + (alpha)*imgs)*255.

if __name__ == '__main__':
    input_tensor = torch.randn(10, 3, 28, 28)
    output_tensor = random_convolution_4d(input_tensor)
    print(output_tensor.shape)
