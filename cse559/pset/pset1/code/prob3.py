## Default modules imported. Import more if you need to.

import numpy as np
from skimage.io import imread, imsave
from scipy.signal import convolve2d as conv2

# Different thresholds to try
T0 = 0.5
T1 = 1.0
T2 = 1.5


########### Fill in the functions below

# Return magnitude, theta of gradients of X
def grads(X):
    #placeholder
    # H = np.zeros(X.shape,dtype=np.float32)
    # theta = np.zeros(X.shape,dtype=np.float32)

    dx = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
    dy = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])

    Ix = conv2(X, dx, 'same', 'symm')
    Iy = conv2(X, dy, 'same', 'symm')

    H = np.sqrt(Ix**2+Iy**2)
    theta = np.arctan2(Iy, Ix)

    return H, theta

def nms(E,H,theta):

    #Round theta
    theta[(theta >= (0 * np.pi / 4)) & (theta <= (0.4 * (1 * np.pi / 4 - 0 * np.pi / 4) + 0 * np.pi / 4))] = 0 * np.pi / 4
    theta[(theta > (0.4 * (1 * np.pi / 4 - 0 * np.pi / 4) + 0 * np.pi / 4)) & (theta <= (0.4 * (2 * np.pi / 4 - 1 * np.pi / 4) + 1 * np.pi / 4))] = 1 * np.pi / 4
    theta[(theta > (0.4 * (2 * np.pi / 4 - 1 * np.pi / 4) + 1 * np.pi / 4)) & (theta <= (0.4 * (3 * np.pi / 4 - 2 * np.pi / 4) + 2 * np.pi / 4))] = 2 * np.pi / 4
    theta[(theta > (0.4 * (3 * np.pi / 4 - 2 * np.pi / 4) + 2 * np.pi / 4)) & (theta <= (0.4 * (4 * np.pi / 4 - 3 * np.pi / 4) + 3 * np.pi / 4))] = 3 * np.pi / 4
    theta[(theta > (0.4 * (4 * np.pi / 4 - 3 * np.pi / 4) + 3 * np.pi / 4)) & (theta <= (4 * np.pi / 4))] = 4 * np.pi / 4

    theta[(theta <= (-0 * np.pi / 4)) & (theta >= (-0.4 * (1 * np.pi / 4 - 0 * np.pi / 4) - 0 * np.pi / 4))] = -0 * np.pi / 4
    theta[(theta < (-0.4 * (1 * np.pi / 4 - 0 * np.pi / 4) - 0 * np.pi / 4)) & (theta >= (-0.4 * (2 * np.pi / 4 - 1 * np.pi / 4) - 1 * np.pi / 4))] = -1 * np.pi / 4
    theta[(theta < (-0.4 * (2 * np.pi / 4 - 1 * np.pi / 4) - 1 * np.pi / 4)) & (theta >= (-0.4 * (3 * np.pi / 4 - 2 * np.pi / 4) - 2 * np.pi / 4))] = -2 * np.pi / 4
    theta[(theta < (-0.4 * (3 * np.pi / 4 - 2 * np.pi / 4) - 2 * np.pi / 4)) & (theta >= (-0.4 * (4 * np.pi / 4 - 3 * np.pi / 4) - 3 * np.pi / 4))] = -3 * np.pi / 4
    theta[(theta < (-0.4 * (4 * np.pi / 4 - 3 * np.pi / 4) - 3 * np.pi / 4)) & (theta >= (-4 * np.pi / 4))] = -4 * np.pi / 4

    E_out = np.zeros(shape=[E.shape[0], E.shape[1]])

    for i in range(1, E.shape[0]-1):
        for j in range(1, E.shape[1]-1):

            if E[i, j] == 1 :

                if (theta[i, j] == 0) | (theta[i, j] == 4 * np.pi / 4) | (theta[i, j] == -4 * np.pi / 4):
                    if (H[i, j] > H[i+1, j]) & (H[i, j] > H[i-1, j]):
                        E_out[i, j] = 1

                if (theta[i, j] == 1 * np.pi / 4) | (theta[i, j] == -3 * np.pi / 4):
                    if (H[i, j] > H[i+1, j+1]) & (H[i, j] > H[i-1, j-1]):
                        E_out[i, j] = 1

                if (theta[i, j] == 2 * np.pi / 4) | (theta[i, j] == -2 * np.pi / 4):
                    if (H[i, j] > H[i, j+1]) & (H[i, j] > H[i, j-1]):
                        E_out[i, j] = 1

                if (theta[i, j] == 3 * np.pi / 4) | (theta[i, j] == -1 * np.pi / 4):
                    if (H[i, j] > H[i-1, j+1]) & (H[i, j] > H[i+1, j-1]):
                        E_out[i, j] = 1

    return E_out

########################## Support code below

from os.path import normpath as fn # Fixes window/linux path conventions
import warnings
warnings.filterwarnings('ignore')

img = np.float32(imread(fn('inputs/p3_inp.jpg')))/255.

H,theta = grads(img)

# imsave(fn('outputs/prob3_a.jpg'),H/(np.max(H[:])))

## Part b

E0 = np.float32(H > T0)
E1 = np.float32(H > T1)
E2 = np.float32(H > T2)

imsave(fn('outputs/prob3_b_0.jpg'),E0)
imsave(fn('outputs/prob3_b_1.jpg'),E1)
imsave(fn('outputs/prob3_b_2.jpg'),E2)

E0n = nms(E0,H,theta)
E1n = nms(E1,H,theta)
E2n = nms(E2,H,theta)

imsave(fn('outputs/prob3_b_nms0.jpg'),E0n)
imsave(fn('outputs/prob3_b_nms1.jpg'),E1n)
imsave(fn('outputs/prob3_b_nms2.jpg'),E2n)
