## Default modules imported. Import more if you need to.

import numpy as np
from skimage.io import imread, imsave


## Fill out these functions yourself

## Take color image, and return 'white balanced' color image
## based on gray world, as described in Problem 2(a). For each
## channel, find the average intensity across all pixels.
##
## Now multiply each channel by multipliers that are inversely
## proportional to these averages, but add upto 3.
def balance2a(img):

    img_output = img.copy()
    alpha_r = 1 / np.mean(img_output[:, :, 0])
    alpha_g = 1 / np.mean(img_output[:, :, 1])
    alpha_b = 1 / np.mean(img_output[:, :, 2])
    alpha_sum = alpha_r + alpha_g + alpha_b

    alpha_r = alpha_r / alpha_sum * 3
    alpha_g = alpha_g / alpha_sum * 3
    alpha_b = alpha_b / alpha_sum * 3

    img_output[:, :, 0] = img_output[:, :, 0] * alpha_r
    img_output[:, :, 1] = img_output[:, :, 1] * alpha_g
    img_output[:, :, 2] = img_output[:, :, 2] * alpha_b

    return img_output


## Take color image, and return 'white balanced' color image
## based on description in Problem 2(b). In each channel, find
## top 10% of the brightest intensities, take their average.
##
## Now multiply each channel by multipliers that are inversely
## proportional to these averages, but add upto 3.
def balance2b(img):

    img_output = img.copy()

    percent_elements_sizes = int(img_output.shape[0] * img_output.shape[1])

    img_output_r_flatten = np.sort(img_output[:, :, 0].flatten())
    img_output_g_flatten = np.sort(img_output[:, :, 1].flatten())
    img_output_b_flatten = np.sort(img_output[:, :, 2].flatten())

    alpha_r = 1 / np.mean(img_output_r_flatten[int(percent_elements_sizes * 0.9):])
    alpha_g = 1 / np.mean(img_output_g_flatten[int(percent_elements_sizes * 0.9):])
    alpha_b = 1 / np.mean(img_output_b_flatten[int(percent_elements_sizes * 0.9):])

    alpha_sum = alpha_r + alpha_g + alpha_b

    alpha_r = alpha_r / alpha_sum * 3
    alpha_g = alpha_g / alpha_sum * 3
    alpha_b = alpha_b / alpha_sum * 3

    img_output[:, :, 0] = img_output[:, :, 0] * alpha_r
    img_output[:, :, 1] = img_output[:, :, 1] * alpha_g
    img_output[:, :, 2] = img_output[:, :, 2] * alpha_b

    return img_output



########################## Support code below

from os.path import normpath as fn # Fixes window/linux path conventions
import warnings
warnings.filterwarnings('ignore')


# Utility functions to clip intensities b/w 0 and 1
# Otherwise imsave complains
def clip(im):
    return np.maximum(0.,np.minimum(1.,im))



############# Main Program
im1 = np.float32(imread(fn('inputs/CC/ex1.jpg')))/255.
im2 = np.float32(imread(fn('inputs/CC/ex2.jpg')))/255.
im3 = np.float32(imread(fn('inputs/CC/ex3.jpg')))/255.


im1a = balance2a(im1)
im2a = balance2a(im2)
im3a = balance2a(im3)

imsave(fn('outputs/prob2a_1.png'),clip(im1a))
imsave(fn('outputs/prob2a_2.png'),clip(im2a))
imsave(fn('outputs/prob2a_3.png'),clip(im3a))

im1b = balance2b(im1)
im2b = balance2b(im2)
im3b = balance2b(im3)

imsave(fn('outputs/prob2b_1.png'),clip(im1b))
imsave(fn('outputs/prob2b_2.png'),clip(im2b))
imsave(fn('outputs/prob2b_3.png'),clip(im3b))
