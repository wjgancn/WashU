## Default modules imported. Import more if you need to.

import numpy as np
from skimage.io import imread, imsave


## Fill out these functions yourself

# Copy from Pset1/Prob6 
def im2wv(img,nLev):

    img_cur = img

    for k in range(nLev):

        img_shape = img_cur.shape
        l = np.zeros(shape=[int(img_shape[0] / 2), int(img_shape[1] / 2)])
        h1 = np.zeros(shape=[int(img_shape[0] / 2), int(img_shape[1] / 2)])
        h2 = np.zeros(shape=[int(img_shape[0] / 2), int(img_shape[1] / 2)])
        h3 = np.zeros(shape=[int(img_shape[0] / 2), int(img_shape[1] / 2)])

        for i in range(0, img_shape[0], 2):
            for j in range(0, img_shape[1], 2):
                a = img_cur[i][j+1]
                b = img_cur[i][j]
                c = img_cur[i+1][j+1]
                d = img_cur[i+1][j]

                l[int(i / 2), int(j / 2)] = (a + b + c + d) / 2
                h1[int(i / 2), int(j / 2)] = (b + d - a - c) / 2
                h2[int(i / 2), int(j / 2)] = (c + d - a - b) / 2
                h3[int(i / 2), int(j / 2)] = (a + d - b - c) / 2

        if k == 0:
            img = [[h1, h2, h3]]
        else:
            img.append([h1, h2, h3])

        img_cur = l

    img.append(l)

    return img


# Copy from Pset1/Prob6 
def wv2im(pyr):

    nLev = pyr.__len__() - 1

    count = nLev  # use to exact data from pyr

    l = pyr[count]
    count -= 1

    [h1, h2, h3] = pyr[count]
    count -= 1

    for k in range(nLev):

        l_shape = l.shape
        img_cur = np.zeros(shape=[l.shape[0] * 2, l.shape[1] * 2])

        for i in range(0, l_shape[0]):
            for j in range(0, l_shape[1]):
                a = 0.5 * l[i, j] - 0.5 * h1[i, j] - 0.5 * h2[i, j] + 0.5 * h3[i, j]
                b = 0.5 * l[i, j] + 0.5 * h1[i, j] - 0.5 * h2[i, j] - 0.5 * h3[i, j]
                c = 0.5 * l[i, j] - 0.5 * h1[i, j] + 0.5 * h2[i, j] - 0.5 * h3[i, j]
                d = 0.5 * l[i, j] + 0.5 * h1[i, j] + 0.5 * h2[i, j] + 0.5 * h3[i, j]

                img_cur[i * 2, j * 2] = b
                img_cur[i * 2 + 1, j * 2] = d
                img_cur[i * 2, j * 2 + 1] = a
                img_cur[i * 2 + 1, j * 2 + 1] = c

        if k != nLev - 1:
            l = img_cur
            [h1, h2, h3] = pyr[count]
            count -= 1

    return img_cur


# Fill this out
# You'll get a numpy array/image of coefficients y
# Return corresponding coefficients x (same shape/size)
# that minimizes (x - y)^2 + lmbda * abs(x)
def denoise_coeff(y,lmbda):

    return np.maximum(np.abs(y) - 0.5*lmbda, 0) * np.sign(y)



########################## Support code below

from os.path import normpath as fn # Fixes window/linux path conventions
import warnings
warnings.filterwarnings('ignore')


# Utility functions to clip intensities b/w 0 and 1
# Otherwise imsave complains
def clip(im):
    return np.maximum(0.,np.minimum(1.,im))



############# Main Program

lmain = 0.88

img = np.float32(imread(fn('inputs/p1.png')))/255.

pyr = im2wv(img,4)
for i in range(len(pyr)-1):
    for j in range(2):
        pyr[i][j] = denoise_coeff(pyr[i][j],lmain/(2**i))
    pyr[i][2] = denoise_coeff(pyr[i][2],np.sqrt(2)*lmain/(2**i))
    
im = wv2im(pyr)        
imsave(fn('outputs/prob1.png'),clip(im))
