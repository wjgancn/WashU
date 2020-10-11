## Default modules imported. Import more if you need to.

import numpy as np
from skimage.io import imread, imsave

# Fill this out
# X is input 8-bit grayscale image
# Return equalized image with intensities from 0-255
def histeq(X):
    from time import time

    init_time = time()

    l = 256
    x_elements = np.unique(X.flatten())
    x_histogram = np.zeros(shape=[l])

    for i in range(x_elements.shape[0]):
        x_histogram[x_elements[i]-1] = np.argwhere(X == x_elements[i]).shape[0]

    x_cdf = np.zeros(shape=[l])

    sum_current = x_histogram[0]
    x_cdf[0] = sum_current

    for i in range(1, l):
        sum_current += x_histogram[i]
        x_cdf[i] = sum_current

    y_cdf = (l-1)*x_cdf/(X.shape[0] * X.shape[1])

    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            X[i, j] = y_cdf[X[i, j]]

    print('Time Consumption: [%.3f] Second ' % (time() - init_time))
    return X
    

########################## Support code below

from os.path import normpath as fn # Fixes window/linux path conventions
import warnings
warnings.filterwarnings('ignore')

img = imread(fn('inputs/p2_inp.jpg'))

out = histeq(img)

out = np.maximum(0,np.minimum(255,out))
out = np.uint8(out)
imsave(fn('outputs/prob2.jpg'),out)
