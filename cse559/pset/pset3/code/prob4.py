## Default modules imported. Import more if you need to.

import numpy as np


## Fill out these functions yourself

# Fits a homography between pairs of pts
#   pts: Nx4 array of (x,y,x',y') pairs of N >= 4 points
# Return homography that maps from (x,y) to (x',y')
#
# Can use np.linalg.svd
def getH(pts):

    A = np.zeros(shape=[3*pts.shape[0], 9])

    for i in range(pts.shape[0]):

        px = pts[i, 0]
        py = pts[i, 1]
        px_dot = pts[i, 2]
        py_dot = pts[i, 3]

        A_i = np.matrix([[0, -1, py_dot], [1, 0, -1*px_dot],[-1*py_dot, px_dot, 0]]) * \
              np.matrix([[px, py, 1, 0, 0, 0, 0, 0, 0], [0, 0, 0, px, py, 1, 0, 0, 0], [0, 0, 0, 0, 0, 0, px, py, 1]])

        A[3*i:3*(i+1), :] = A_i

    A = np.matrix(A)
    h = np.linalg.svd(A)[2][8] # Exact the last row
    h.shape = [3, 3]

    return np.matrix(h)
    

# Splices the source image into a quadrilateral in the dest image,
# where dpts in a 4x2 image with each row giving the [x,y] co-ordinates
# of the corner points of the quadrilater (in order, top left, top right,
# bottom left, and bottom right).
#
# Note that both src and dest are color images.
#
# Return a spliced color image.
def splice(src,dest,dpts):

    w = src.shape[0]
    h = src.shape[1]
    # w = 10
    # h = 10

    pts = np.zeros(shape=[4, 4])
    pts[0, 0] = 0
    pts[0, 1] = 0
    pts[0, 2:4] = dpts[0]

    pts[1, 0] = w-1
    pts[1, 1] = 0
    pts[1, 2:4] = dpts[1]

    pts[2, 0] = 0
    pts[2, 1] = h-1
    pts[2, 2:4] = dpts[2]

    pts[3, 0] = w-1
    pts[3, 1] = h-1
    pts[3, 2:4] = dpts[3]

    # print(getH(pts))
    h_mat = getH(pts)

    src_cor = np.matrix(np.zeros(shape=[3, w*h]))
    for i in range(w):
        src_cor[0, i * h:(i + 1) * h] = i
        src_cor[1, i * h:(i + 1) * h] = np.linspace(0, h-1, h)
        src_cor[2, i * h:(i + 1) * h] = 1

    dst_cor = np.array(h_mat * src_cor)
    dst_cor = dst_cor / dst_cor[2, :]

    dst_cor = dst_cor.astype(dtype=np.uint16)
    src_cor = src_cor.astype(dtype=np.uint16)

    for i in range(w*h):
        dest[dst_cor[0, i], dst_cor[1, i]] = src[src_cor[0, i], src_cor[1, i]]

    return dest
    
    
########################## Support code below

from skimage.io import imread, imsave
from os.path import normpath as fn # Fixes window/linux path conventions
import warnings
warnings.filterwarnings('ignore')


simg = np.float32(imread(fn('inputs/p4src.png')))/255.
dimg = np.float32(imread(fn('inputs/p4dest.png')))/255.
dpts = np.float32([ [276,54],[406,79],[280,182],[408,196]]) # Hard coded

comb = splice(simg,dimg,dpts)

imsave(fn('outputs/prob4.png'),comb)
