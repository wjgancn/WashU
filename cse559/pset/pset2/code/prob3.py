### Default modules imported. Import more if you need to.
### DO NOT USE linalg.lstsq from numpy or scipy

import numpy as np
from skimage.io import imread, imsave

## Fill out these functions yourself


# Inputs:
#    imgs: A list of N color images, each of which is HxWx3
#    L:    An Nx3 matrix where each row corresponds to light vector
#          for corresponding image.
#    mask: A 0-1 mask of size HxW, showing where observed data is 'valid'.
#
# Returns nrm:
#    nrm: HxWx3 Unit normal vector at each location.
#
# Be careful about division by zero at mask==0 for normalizing unit vectors.
def pstereo_n(imgs, L, mask):

    I = np.zeros(shape=[imgs[0].shape[0], imgs[0].shape[1], imgs.__len__(), 1])
    n = np.zeros(shape=[imgs[0].shape[0], imgs[0].shape[1], 3, 1])


    for i in range(imgs.__len__()):
        I[:, :, i, 0] = (imgs[i][:,:,0] + imgs[i][:,:,1] + imgs[i][:,:,2])

    Ltl = np.array(np.matrix(L.transpose()) * np.matrix(L))

    for i in range(n.shape[0]):
        for j in range(n.shape[1]):

            if mask[i, j] == 1:
                b = np.matrix(L.transpose()) * np.matrix(I[i, j, :, :])
                n_current = np.array(np.linalg.solve(Ltl, b))
                n[i, j, :, :] = n_current / np.sum(n_current)

        if i % 50 == 0:
            print("pstereo_n: [%.2f] Percentage" % ((i+1)*100 / n.shape[0]))

    # import matplotlib.pyplot as plt
    # plt.imshow(I[:, :, 2], cmap='gray')
    # plt.show()
    n.shape = [n.shape[0], n.shape[1], 3]

    return n


# Inputs:
#    imgs: A list of N color images, each of which is HxWx3
#    nrm:  HxWx3 Unit normal vector at each location (from pstereo_n)
#    L:    An Nx3 matrix where each row corresponds to light vector
#          for corresponding image.
#    mask: A 0-1 mask of size HxW, showing where observed data is 'valid'.
#
# Returns alb:
#    alb: HxWx3 RGB Color Albedo values
#
# Be careful about division by zero at mask==0.
def pstereo_alb(imgs, nrm, L, mask):

    alb = np.zeros([imgs[0].shape[0], imgs[0].shape[1], 3])

    R =  np.zeros([imgs[0].shape[0], imgs[0].shape[1], imgs.__len__()])
    G =  np.zeros([imgs[0].shape[0], imgs[0].shape[1], imgs.__len__()])
    B = np.zeros([imgs[0].shape[0], imgs[0].shape[1], imgs.__len__()])

    for i in range(imgs.__len__()):
        R[:, :, i] = imgs[i][:, :, 0]
        G[:, :, i] = imgs[i][:, :, 1]
        B[:, :, i] = imgs[i][:, :, 2]

    l = np.matrix(L)

    for i in range(alb.shape[0]):
        for j in range(alb.shape[1]):

            if mask[i, j] == 1:
                X = l * np.matrix(nrm[i, j, :]).transpose()
                XtX_1Xt = (X.transpose() * X) ** -1 * X.transpose()

                y_r = R[i, j, :]
                y_r.shape = [y_r.shape[0], 1]
                y_r = np.matrix(y_r)

                alb[i, j, 0] = XtX_1Xt * y_r

                y_g = G[i, j, :]
                y_g.shape = [y_g.shape[0], 1]
                y_g = np.matrix(y_g)

                alb[i, j, 1] = XtX_1Xt * y_g

                y_b = B[i, j, :]
                y_b.shape = [y_b.shape[0], 1]
                y_b = np.matrix(y_b)

                alb[i, j, 2] = XtX_1Xt * y_b

        if i % 50 == 0:
            print("pstereo_alb: [%.2f] Percentage" % ((i+1)*100 / alb.shape[0]))

    return alb
    
########################## Support code below

from os.path import normpath as fn # Fixes window/linux path conventions
import warnings
warnings.filterwarnings('ignore')

### Light directions matrix
L = np.float32( \
                [[  4.82962877e-01,   2.58819044e-01,   8.36516321e-01],
                 [  2.50000030e-01,   2.58819044e-01,   9.33012664e-01],
                 [ -4.22219593e-08,   2.58819044e-01,   9.65925813e-01],
                 [ -2.50000000e-01,   2.58819044e-01,   9.33012664e-01],
                 [ -4.82962966e-01,   2.58819044e-01,   8.36516261e-01],
                 [ -5.00000060e-01,   0.00000000e+00,   8.66025388e-01],
                 [ -2.58819044e-01,   0.00000000e+00,   9.65925813e-01],
                 [ -4.37113883e-08,   0.00000000e+00,   1.00000000e+00],
                 [  2.58819073e-01,   0.00000000e+00,   9.65925813e-01],
                 [  4.99999970e-01,   0.00000000e+00,   8.66025448e-01],
                 [  4.82962877e-01,  -2.58819044e-01,   8.36516321e-01],
                 [  2.50000030e-01,  -2.58819044e-01,   9.33012664e-01],
                 [ -4.22219593e-08,  -2.58819044e-01,   9.65925813e-01],
                 [ -2.50000000e-01,  -2.58819044e-01,   9.33012664e-01],
                 [ -4.82962966e-01,  -2.58819044e-01,   8.36516261e-01]])


# Utility functions to clip intensities b/w 0 and 1
# Otherwise imsave complains
def clip(im):
    return np.maximum(0.,np.minimum(1.,im))


############# Main Program


# Load image data
imgs = []
for i in range(L.shape[0]):
    imgs = imgs + [np.float32(imread(fn('inputs/phstereo/img%02d.png' % i)))/255.]

mask = np.float32(imread(fn('inputs/phstereo/mask.png')) > 0)

nrm = pstereo_n(imgs,L,mask)

nimg = nrm/2.0+0.5
nimg = clip(nimg * mask[:,:,np.newaxis])
imsave(fn('outputs/prob3_nrm.png'),nimg)


alb = pstereo_alb(imgs,nrm,L,mask)

alb = alb / np.max(alb[:])
alb = clip(alb * mask[:,:,np.newaxis])

imsave(fn('outputs/prob3_alb.png'),alb)
