## Default modules imported. Import more if you need to.

import numpy as np
from skimage.io import imread, imsave
from scipy.signal import convolve2d as conv2


# Fill this out
def kernpad(K,size):

    Ko = np.zeros(size,dtype=np.float32)

    K_center = int((K.shape[0] - 1)/2)
    K_up_left = K[:K_center, :K_center]
    K_down_right= K[K_center:, K_center:]

    K_up_right = K[:K_center, K_center:]
    K_down_left = K[K_center:, :K_center]

    Ko[:K_down_right.shape[0], :K_down_right.shape[1]] = K_down_right
    Ko[-1 * K_up_left.shape[0]:, -1 * K_up_left.shape[1]:] = K_up_left
    Ko[-1 * K_up_right.shape[0]:, : K_up_right.shape[1]] = K_up_right
    Ko[:K_down_left.shape[0],  -1*K_down_left.shape[1]:] = K_down_left

    # import matplotlib.pyplot as plt
    # plt.imshow(Ko, cmap='gray')
    # plt.show()

    return Ko

########################## Support code below

from os.path import normpath as fn # Fixes window/linux path conventions
import warnings
warnings.filterwarnings('ignore')

img = np.float32(imread(fn('inputs/p5_inp.jpg')))/255.

# Create Gaussian Kernel
x = np.float32(range(-21,22))
x,y = np.meshgrid(x,x)
G = np.exp(-(x*x+y*y)/2/9.)
G = G / np.sum(G[:])


# Traditional convolve
v1 = conv2(img,G,'same','wrap')

# Convolution in Fourier domain
G = kernpad(G,img.shape)
v2f = np.fft.fft2(G)*np.fft.fft2(img)
v2 = np.real(np.fft.ifft2(v2f))

# Stack them together and save
out = np.concatenate([img,v1,v2],axis=1)
out = np.minimum(1.,np.maximum(0.,out))

imsave(fn('outputs/prob5.jpg'),out)


                 
