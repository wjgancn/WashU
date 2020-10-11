## Default modules imported. Import more if you need to.

import numpy as np


#########################################
### Hamming distance computation
### You can call the function hamdist with two
### uint32 bit arrays of the same size. It will
### return another array of the same size with
### the elmenet-wise hamming distance.
hd8bit = np.zeros((256,))
for i in range(256):
    v = i
    for k in range(8):
        hd8bit[i] = hd8bit[i] + v%2
        v=v//2


def hamdist(x,y):
    dist = np.zeros(x.shape)
    g = x^y
    for i in range(4):
        dist = dist + hd8bit[g%256]
        g = g // 256
    return dist
#########################################





## Fill out these functions yourself

# Compute a 5x5 census transform of the grayscale image img.
# Return a uint32 array of the same shape
def census(img):

    W = img.shape[1]
    H = img.shape[0]
    c = np.zeros([H,W],dtype=np.uint32)

    filter_images = np.zeros([H+4, W+4, 25])
    # print(filter_images[2:-2, 2:-2, 0].shape)

    filter_counts = 0
    for i in range(5):
        for j in range(5):
            filter_images[i:i+H, j:j+W, filter_counts] = img
            filter_counts += 1

    binary_count = 0
    for i in range(25):
        if i != 12: # do not compare (x,y) with itself
            comparision = img - filter_images[2:-2, 2:-2, i]
            # print(comparision.max())
            comparision[comparision >= 0] = 1
            comparision[comparision < 0] = 0
            c[:, :] = c[:, :] + comparision*(2**binary_count) ## \Sum 2^(binary_count) = decimal
            binary_count += 1

    return c
    

# Given left and right image and max disparity D_max, return a disparity map
# based on matching with  hamming distance of census codes. Use the census function
# you wrote above.
#
# d[x,y] implies that left[x,y] matched best with right[x-d[x,y],y]. Disparity values
# should be between 0 and D_max (both inclusive).
def smatch(left,right,dmax):

    H = left.shape[0]
    W = left.shape[1]
    c = np.zeros(shape=[H, W - dmax, dmax + 1]) # only consider when dmax <= x

    right_census = census(right)
    left_census = census(left)

    for i in range(dmax + 1):

        c[:, :, i] = hamdist(right_census[:, dmax - i: W - i], left_census[:, dmax : W])

    d = np.argmin(c, axis=2)

    return d
    
    
########################## Support code below

from skimage.io import imread, imsave
from os.path import normpath as fn # Fixes window/linux path conventions
import matplotlib.cm as cm
import warnings
warnings.filterwarnings('ignore')


left = imread(fn('inputs/left.jpg'))
right = imread(fn('inputs/right.jpg'))

d = smatch(left,right,40)

# Map to color and save
dimg = cm.jet(np.minimum(1,np.float32(d.flatten())/20.))[:,0:3]
dimg = dimg.reshape([d.shape[0],d.shape[1],3])
imsave(fn('outputs/prob5.png'),dimg)
