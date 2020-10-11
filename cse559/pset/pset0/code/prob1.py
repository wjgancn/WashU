## Default modules imported. Import more if you need to.
import numpy as np
from skimage.io import imread, imsave

# Edit the following two functions

def vflip(X):
    X = X[::-1,:,:]
    return X


def hflip(X):
    X = X[:,::-1,:]
    return X


########################## Support code below

from os.path import normpath as fn # Fixes window/linux path conventions
import warnings
warnings.filterwarnings('ignore')

img = imread(fn('inputs/cat.jpg'))

flipy = vflip(img)
flipx = hflip(img)

imsave(fn('outputs/flipy.jpg'),flipy)
imsave(fn('outputs/flipx.jpg'),flipx)
