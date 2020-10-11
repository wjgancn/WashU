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


# Copy this from solution to problem 2.
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


# Implement the forward-backward viterbi method to smooth
# only along horizontal lines. Assume smoothness cost of
# 0 if disparities equal, P1 if disparity difference <= 1, P2 otherwise.
#
# Function takes in cost volume cv, and values of P1 and P2
# Return the disparity map
def viterbilr(cv,P1,P2):
    H = cv.shape[0]
    W = cv.shape[1]
    D = cv.shape[2]

    d = np.zeros(shape=[H, W], dtype=np.int16)
    z = np.zeros(shape=[H, W, D], dtype=np.float32)

    S = np.abs(np.arange(0, D).reshape([D, 1]) - np.arange(0, D).reshape([1, D])).astype(np.float32)
    mask_p1 = S == 1
    mask_p2 = S > 1
    S[mask_p1] = P1
    S[mask_p2] = P2

    c_overline = cv.copy()

    for i in range(1, W):

        src = c_overline[:, i-1, :]

        z[:, i, :] = np.argmin(src[:, np.newaxis, :] + S[np.newaxis, :, :], axis= -1 )
        c_overline[:, i, :] = c_overline[:, i, :] + np.min(src[:, np.newaxis, :] + S[np.newaxis, :, :], axis= -1 )

        if i % 50 == 0:
            print("Forward: [%.3f] Percentage" % (100*i/(W-2)))


    d[:, W - 1] = np.argmin(c_overline[:,  W-1, :], axis = -1 )

    for i in range(W - 2):

        index = W - 2 - i

        d[:, index] = z[np.arange(0, H), index + 1, d[:, index + 1]]

        if i % 50 == 0:
            print("Backward: [%.3f] Percentage" % (100*i/(W-2)))

    return d


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
                   
cv = buildcv(left_g,right_g,50)
d = viterbilr(cv,0.5,16)

# Map to color and save
dimg = cm.jet(np.minimum(1,np.float32(d.flatten())/50.))[:,0:3]
dimg = dimg.reshape([d.shape[0],d.shape[1],3])
imsave(fn('outputs/prob3a.jpg'),dimg)
