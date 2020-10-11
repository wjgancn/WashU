## Default modules imported. Import more if you need to.
import numpy as np
from scipy.signal import convolve2d as conv2

# Use these as the x and y derivative filters
fx = np.float32([[1,0,-1]]) * np.float32([[1,1,1]]).T / 6.
fy = fx.T


# Compute optical flow using the lucas kanade method
# Use the fx, fy, defined above as the derivative filters
# and compute derivatives on the average of the two frames.
# Also, consider (x',y') values in a WxW window.
# Return two image shape arrays u,v corresponding to the
# horizontal and vertical flow.
def lucaskanade(f1,f2,W):
    u = np.zeros(f1.shape)
    v = np.zeros(f1.shape)

    avg_f = ( f1 + f2 ) / 2

    Ix = conv2(avg_f, fx, 'same')
    Iy = conv2(avg_f, fy, 'same')
    It = f2 - f1

    Ix2 = Ix ** 2
    Iy2 = Iy ** 2
    IxIy = Ix * Iy
    IxIt = Ix * It
    IyIt = Iy * It

    local_sum = np.ones(shape=[W, W])
    Ix2 = conv2(Ix2, local_sum, 'same')
    Iy2 = conv2(Iy2, local_sum, 'same')
    IxIy = conv2(IxIy, local_sum, 'same')
    IxIt = conv2(IxIt, local_sum, 'same')
    IyIt = conv2(IyIt, local_sum, 'same')

    lm = np.zeros([f1.shape[0], f1.shape[1], 2, 2])
    lm[:, :, 0, 0] = Ix2 + 0.01
    lm[:, :, 0, 1] = IxIy
    lm[:, :, 1, 0] = IxIy
    lm[:, :, 1, 1] = Iy2 + 0.01

    rm = np.zeros([f1.shape[0], f1.shape[1], 2, 1])
    rm[:, :, 0, 0] = IxIt * -1
    rm[:, :, 1, 0] = IyIt * -1

    mul = np.matmul(np.linalg.inv(lm), rm)

    u[:, :] = mul[:, :, 0, 0]
    v[:, :] = mul[:, :, 1, 0]

    return u,v
    
########################## Support code below

from skimage.io import imread, imsave
from os.path import normpath as fn # Fixes window/linux path conventions
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')


f1 = np.float32(imread(fn('inputs/frame10.jpg')))/255.
f2 = np.float32(imread(fn('inputs/frame11.jpg')))/255.

u,v = lucaskanade(f1,f2,11)


# Display quiver plot by downsampling
x = np.arange(u.shape[1])
y = np.arange(u.shape[0])
x,y = np.meshgrid(x,y[::-1])
plt.quiver(x[::8,::8],y[::8,::8],u[::8,::8],-v[::8,::8],pivot='mid')

plt.show()
