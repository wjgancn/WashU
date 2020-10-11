## Default modules imported. Import more if you need to.

import numpy as np
from skimage.io import imread, imsave


## Fill out these functions yourself

def im2wv(img,nLev):
    # Placeholder that does nothing

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


def wv2im(pyr):
    # Placeholder that does nothing
    # m.I * 2
    # matrix([[ 0.5, -0.5, -0.5,  0.5],
    #         [ 0.5,  0.5, -0.5, -0.5],
    #         [ 0.5, -0.5,  0.5, -0.5],
    #         [ 0.5,  0.5,  0.5,  0.5]])
    #
    # 0.5l - 0.5h1 - 0.5h2 + 0.5h3 = a
    # 0.5l + 0.5h1 - 0.5h2 - 0.5h3 = b
    # 0.5l - 0.5h1 + 0.5h2 - 0.5h3 = c
    # 0.5l + 0.5h1 + 0.5h2 + 0.5h3 = d

    nLev = pyr.__len__() - 1

    count = nLev # use to exact data from pyr

    l = pyr[count]
    count -= 1

    [h1, h2, h3] = pyr[count]
    count -= 1

    for k in range(nLev):

        l_shape = l.shape
        img_cur = np.zeros(shape=[l.shape[0] * 2, l.shape[1] * 2])

        for i in range(0, l_shape[0]):
            for j in range(0, l_shape[1]):

                a = 0.5*l[i,j] - 0.5*h1[i,j] - 0.5*h2[i,j] + 0.5*h3[i,j]
                b = 0.5*l[i,j] + 0.5*h1[i,j] - 0.5*h2[i,j] - 0.5*h3[i,j]
                c = 0.5*l[i,j] - 0.5*h1[i,j] + 0.5*h2[i,j] - 0.5*h3[i,j]
                d = 0.5*l[i,j] + 0.5*h1[i,j] + 0.5*h2[i,j] + 0.5*h3[i,j]

                img_cur[i*2,j*2] = b
                img_cur[i*2+1, j*2] = d
                img_cur[i*2, j*2+1] = a
                img_cur[i*2+1, j*2+1] = c

        if k != nLev - 1 :
            l = img_cur
            [h1, h2, h3] = pyr[count]
            count -= 1

    return img_cur



########################## Support code below

from os.path import normpath as fn # Fixes window/linux path conventions
import warnings
warnings.filterwarnings('ignore')


# Utility functions to clip intensities b/w 0 and 1
# Otherwise imsave complains
def clip(im):
    return np.maximum(0.,np.minimum(1.,im))


# Visualize pyramid like in slides
def vis(pyr, lev=0):
    if len(pyr) == 1:
        return pyr[0]/(2**lev)

    sz=pyr[0][0].shape
    sz1 = [sz[0]*2,sz[1]*2]
    img = np.zeros(sz1,dtype=np.float32)

    img[0:sz[0],0:sz[1]] = vis(pyr[1:],lev+1)

    # Just scale / shift gradient images for visualization
    img[sz[0]:,0:sz[1]] = pyr[0][0]*(2**(1-lev))+0.5
    img[0:sz[0],sz[1]:] = pyr[0][1]*(2**(1-lev))+0.5
    img[sz[0]:,sz[1]:] = pyr[0][2]*(2**(1-lev))+0.5

    return img



############# Main Program


img = np.float32(imread(fn('inputs/p6_inp.jpg')))/255.

# Visualize pyramids
pyr = im2wv(img,1)
imsave(fn('outputs/prob6a_1.jpg'),clip(vis(pyr)))

pyr = im2wv(img,2)
imsave(fn('outputs/prob6a_2.jpg'),clip(vis(pyr)))

pyr = im2wv(img,3)
imsave(fn('outputs/prob6a_3.jpg'),clip(vis(pyr)))

# Inverse transform to reconstruct image
im = clip(wv2im(pyr))
imsave(fn('outputs/prob6b.jpg'),im)

# Zero out some levels and reconstruct
for i in range(len(pyr)-1):

    for j in range(3):
        pyr[i][j][...] = 0.

    im = clip(wv2im(pyr))
    imsave(fn('outputs/prob6b_%d.jpg' % i),im)
