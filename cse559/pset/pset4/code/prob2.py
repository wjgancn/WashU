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


# Given left and right grayscale images and max disparity D_max, build a HxWx(D_max+1) array
# corresponding to the cost volume. For disparity d where x-d < 0, fill a cost
# value of 24 (the maximum possible hamming distance).
#
# You can call the hamdist function above, and copy your census function from the
# previous problem set.

def buildcv(left,right,dmax):

    def census(img):

        W = img.shape[1]
        H = img.shape[0]
        c = np.zeros([H, W], dtype=np.uint32)

        filter_images = np.zeros([H + 4, W + 4, 25])
        # print(filter_images[2:-2, 2:-2, 0].shape)

        filter_counts = 0
        for i in range(5):
            for j in range(5):
                filter_images[i:i + H, j:j + W, filter_counts] = img
                filter_counts += 1

        binary_count = 0
        for i in range(25):
            if i != 12:  # do not compare (x,y) with itself
                comparision = img - filter_images[2:-2, 2:-2, i]
                # print(comparision.max())
                comparision[comparision >= 0] = 1
                comparision[comparision < 0] = 0
                c[:, :] = c[:, :] + comparision * (2 ** binary_count)  ## \Sum 2^(binary_count) = decimal
                binary_count += 1

        return c

    H = left.shape[0]
    W = left.shape[1]
    c = 24 * np.ones(shape=[H, W, dmax + 1], dtype=np.float32)  # only consider when dmax <= x

    right_census = census(right)
    left_census = census(left)

    for i in range(dmax + 1):
        c[:, dmax:, i] = hamdist(right_census[:, dmax - i: W - i], left_census[:, dmax: W])

    return c


# Fill this out
# CV is the cost-volume to be filtered.
# X is the left color image that will serve as guidance.
# K is the support of the filter (2K+1)x(2K+1)
# sgm_s is std of spatial gaussian
# sgm_i is std of intensity gaussian
#
# Feel free to look at the solution key for bfilt function form problem set 1.
def bfilt(cv,X,K,sgm_s,sgm_i):

    H = X.shape[0]
    W = X.shape[1]

    yy = np.zeros(cv.shape)
    B = np.zeros([H, W, 1])

    for y in range(-K, K + 1):

        for x in range(-K, K + 1):

            if y < 0:

                y1a = 0
                y1b = -y
                y2a = H + y
                y2b = H

            else:

                y1a = y
                y1b = 0
                y2a = H
                y2b = H - y

            if x < 0:

                x1a = 0
                x1b = -x
                x2a = W + x
                x2b = W

            else:

                x1a = x
                x1b = 0
                x2a = W
                x2b = W - x

            bxy = X[y1a:y2a, x1a:x2a, :] - X[y1b:y2b, x1b:x2b, :]
            bxy = np.sum(bxy * bxy, axis=2, keepdims=True)

            bxy = bxy / (sgm_i ** 2) + np.float32(y ** 2 + x ** 2) / (sgm_s ** 2)
            bxy = np.exp(-bxy / 2.0)

            B[y1b:y2b, x1b:x2b, :] = B[y1b:y2b, x1b:x2b, :] + bxy

            yy[y1b:y2b, x1b:x2b, :] = yy[y1b:y2b, x1b:x2b, :] + bxy * cv[y1a:y2a, x1a:x2a, :]

    return yy / B

    
########################## Support code below

from skimage.io import imread, imsave
from os.path import normpath as fn # Fixes window/linux path conventions
import matplotlib.cm as cm
import warnings
warnings.filterwarnings('ignore')


left = np.float32(imread(fn('inputs/left.jpg')))/255.
right = np.float32(imread(fn('inputs/right.jpg')))/255.

left_g = np.mean(left,axis=2)
right_g = np.mean(right,axis=2)
                   
cv0 = buildcv(left_g,right_g,50)

cv1 = bfilt(cv0,left,5,2,0.5)
    

d0 = np.argmin(cv0,axis=2)
d1 = np.argmin(cv1,axis=2)

# Map to color and save
dimg = cm.jet(np.minimum(1,np.float32(d0.flatten())/50.))[:,0:3]
dimg = dimg.reshape([d0.shape[0],d0.shape[1],3])
imsave(fn('outputs/prob2a.jpg'),dimg)

dimg = cm.jet(np.minimum(1,np.float32(d1.flatten())/50.))[:,0:3]
dimg = dimg.reshape([d1.shape[0],d1.shape[1],3])
imsave(fn('outputs/prob2b.jpg'),dimg)
