## Default modules imported. Import more if you need to.

import numpy as np
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
# Implement in Fourier Domain / Frankot-Chellappa
def ntod(nrm, mask, lmda):

    def kernpad(K, size):
        Ko = np.zeros(size, dtype=np.float32)

        K_center = int((K.shape[0] - 1) / 2)
        K_up_left = K[:K_center, :K_center]
        K_down_right = K[K_center:, K_center:]

        K_up_right = K[:K_center, K_center:]
        K_down_left = K[K_center:, :K_center]

        Ko[:K_down_right.shape[0], :K_down_right.shape[1]] = K_down_right
        Ko[-1 * K_up_left.shape[0]:, -1 * K_up_left.shape[1]:] = K_up_left
        Ko[-1 * K_up_right.shape[0]:, : K_up_right.shape[1]] = K_up_right
        Ko[:K_down_left.shape[0], -1 * K_down_left.shape[1]:] = K_down_left

        return Ko

    fx = np.array([[0, 0, 0], [0.5, 0, -0.5], [0, 0, 0]])
    fy = np.array([[0, -0.5, 0], [0, 0, 0], [0, 0.5, 0]])
    fr = np.array([[-1/9, -1/9, -1/9], [-1/9, 8/9, -1/9], [-1/9, -1/9, -1/9]])

    Fx = np.fft.fft2(kernpad(fx, size=mask.shape))
    Fy = np.fft.fft2(kernpad(fy, size=mask.shape))
    Fr = np.fft.fft2(kernpad(fr, size=mask.shape))

    gx = np.zeros(shape=[nrm.shape[0], nrm.shape[1]])
    gy = np.zeros(shape=[nrm.shape[0], nrm.shape[1]])

    for i in range(nrm.shape[0]):
        for j in range(nrm.shape[1]):
            if mask[i, j] == 1:

                gx[i, j] = -1 * nrm[i, j, 0] / nrm[i, j, 2]
                gy[i, j] = -1 * nrm[i, j, 1] / nrm[i, j, 2]

        print("ntod - generate g_x and g_y: [%.2f] Percentage" % ((i + 1) * 100 / nrm.shape[0]))

    Gx = np.fft.fft2(gx)
    Gy = np.fft.fft2(gy)

    Fz = ((Fx * Gx) + (Fy * Gy)) / (Fx**2 + Fy**2 + lmda * (Fr ** 2))
    z = np.real(np.fft.ifft2(Fz))

    # print(z)

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
Z = ntod(nrm,mask,1e-6)


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
