## Default modules imported. Import more if you need to.

import numpy as np
from scipy.signal import convolve2d as conv2
from skimage.io import imread, imsave


## Fill out these functions yourself


# Inputs:
#    nrm: HxWx3. Unit normal vectors at each location. All zeros at mask == 0
#    mask: A 0-1 mask of size HxW, showing where observed data is 'valid'.
#    lmda: Scalar value of lambda to be used for regularizer weight as in slides.
#
# Returns depth map Z of size HxWx3.
#
# Be careful about division by 0.
#
# Implement using conjugate gradient, with a weight = 0 for mask == 0, and proportional
# to n_z^2 elsewhere. See slides.

def ntod(nrm, mask, lmda):

    w = nrm[:, :, 2] ** 2
    w[mask == 0] = 0

    fx = np.array([[0, 0, 0], [0.5, 0, -0.5], [0, 0, 0]])
    fy = np.array([[0, -0.5, 0], [0, 0, 0], [0, 0.5, 0]])
    fr = np.array([[-1 / 9, -1 / 9, -1 / 9], [-1 / 9, 8 / 9, -1 / 9], [-1 / 9, -1 / 9, -1 / 9]])

    n_z = nrm[:, :, 2]
    n_z[mask == 0] = 1 # Avoid divide "0"

    gx = -1 * nrm[:, :, 0] / n_z
    gy = -1 * nrm[:, :, 1] / n_z

    gx[mask == 0] = 0
    gy[mask == 0] = 0

    def Q(x):
        return conv2(conv2(x, fx, mode='same')*w, np.flip(fx), mode='same') + conv2(conv2(x, fy, mode='same')*w, np.flip(fy), mode='same') + lmda*(conv2(conv2(x, fr, mode='same'), np.flip(fr), mode='same'))

    b = conv2(gx * w, np.flip(fx), mode='same') + conv2(gy * w, np.flip(fy), mode='same')
    z = np.zeros(shape = [nrm.shape[0], nrm.shape[1]])
    r = b - Q(z)
    p = r

    for k in range(200):
        r_previous = r
        alpha =np.sum(r*r)/ np.sum(p * Q(p))
        z = z + alpha * p
        r = r - alpha * Q(p)

        beta = np.sum(r * r) / np.sum(r_previous * r_previous)
        p = r + beta * p

        print("ntod: [%d]" % k)

    return z


########################## Support code below

from os.path import normpath as fn # Fixes window/linux path conventions
import warnings
warnings.filterwarnings('ignore')

import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D


#### Main function

nrm = imread(fn('inputs/phstereo/true_normals.png'))

# Un-comment  next line to read your output instead
# nrm = imread(fn('outputs/prob3_nrm.png'))


mask = np.float32(imread(fn('inputs/phstereo/mask.png')) > 0)

nrm = np.float32(nrm/255.0)
nrm = nrm*2.0-1.0
nrm = nrm * mask[:,:,np.newaxis]


# Main Call
Z = ntod(nrm,mask,1e-7)


# Plot 3D shape

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

x,y = np.meshgrid(np.float32(range(nrm.shape[1])),np.float32(range(nrm.shape[0])))
x = x - np.mean(x[:])
y = y - np.mean(y[:])

Zmsk = Z.copy()
Zmsk[mask == 0] = np.nan
Zmsk = Zmsk - np.nanmedian(Zmsk[:])

lim = 100
ax.plot_surface(x,-y,Zmsk, \
                linewidth=0,cmap=cm.inferno,shade=True,\
                vmin=-lim,vmax=lim)

ax.set_xlim3d(-450,450)
ax.set_ylim3d(-450,450)
ax.set_zlim3d(-450,450)

plt.show()
