## Default modules imported. Import more if you need to.

import numpy as np
from skimage.io import imread, imsave

# Fill this out
# X is input color image
# K is the support of the filter (2K+1)x(2K+1)
# sgm_s is std of spatial gaussian
# sgm_i is std of intensity gaussian
def bfilt(X,K,sgm_s,sgm_i):
    # Placeholder
    from time import time
    init_time = time()

    width, height, channels = X.shape
    center_width = int(width/2)
    center_height = int(height/2)

    X_compen = np.zeros(shape=[width + K*2, height + K*2, channels])
    X_compen[K:-K, K:-K, :] = X.copy()
    X_output = X.copy()

    X_diff = np.zeros(shape=[width, height, channels, K * 2 + 1, K * 2 + 1])
    n_diff = np.zeros(shape=[width, height, channels, K * 2 + 1, K * 2 + 1])

    for i in range(K * 2 + 1):
        for j in range(K * 2 + 1):
            X_diff[:, :, :, i, j] = (X_compen[i:i+width, j:j+height, :] - X)** 2
            n_diff[:, :, :, i, j] = (i-K)**2 + (j-K)**2

    X_diff = X_diff / (2 * (sgm_i ** 2))
    n_diff = n_diff / (2 * (sgm_s ** 2))

    B = np.exp(-1*(n_diff + X_diff))

    B_mask = np.zeros(shape=[width + K*2, height + K*2])
    B_mask[K:-K, K:-K] = 1

    print('Original B', time() - init_time, ' s')

    for i in range(width):
        for j in range(height):
            for c in range(channels):

                B_cur = B[i, j, c, :, :] * B_mask[i:i+2*K+1, j:j+2*K+1]
                B_cur = B_cur / np.sum(B_cur)

                X_output[i, j, c] = np.sum(X_compen[i:i+2*K+1, j:j+2*K+1, c] * B_cur)

    print('ALL Done!', time()-init_time, ' s')

    return X_output


########################## Support code below

def clip(im):
    return np.maximum(0.,np.minimum(1.,im))

from os.path import normpath as fn # Fixes window/linux path conventions
import warnings
warnings.filterwarnings('ignore')

img1 = np.float32(imread(fn('inputs/p4_nz1.jpg')))/255.
img2 = np.float32(imread(fn('inputs/p4_nz2.jpg')))/255.

K=9

print("Creating outputs/prob4_1_a.jpg")
im1A = bfilt(img1,K,2,0.5)
imsave(fn('outputs/prob4_1_a.jpg'),clip(im1A))

print("Creating outputs/prob4_1_b.jpg")
im1B = bfilt(img1,K,4,0.25)
imsave(fn('outputs/prob4_1_b.jpg'),clip(im1B))

print("Creating outputs/prob4_1_c.jpg")
im1C = bfilt(img1,K,16,0.125)
imsave(fn('outputs/prob4_1_c.jpg'),clip(im1C))

# Repeated application
print("Creating outputs/prob4_1_rep.jpg")
im1D = bfilt(img1,K,2,0.125)
for i in range(8):
    im1D = bfilt(im1D,K,2,0.125)
imsave(fn('outputs/prob4_1_rep.jpg'),clip(im1D))

# Try this on image with more noise    
print("Creating outputs/prob4_2_rep.jpg")
im2D = bfilt(img2,2,8,0.125)
for i in range(16):
    im2D = bfilt(im2D,K,2,0.125)
imsave(fn('outputs/prob4_2_rep.jpg'),clip(im2D))
