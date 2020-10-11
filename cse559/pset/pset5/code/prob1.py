## Default modules imported. Import more if you need to.
### Problem designed by Abby Stylianou

import numpy as np
from scipy.signal import convolve2d as conv2

def get_cluster_centers(im,num_clusters):
    # Implement a method that returns an initial grid of cluster centers. You should first
    # create a grid of evenly spaced centers (hint: np.meshgrid), and then use the method
    # discussed in class to make sure no centers are initialized on a sharp boundary.
    # You can use the get_gradients method from the support code below.
    cluster_centers = np.zeros((num_clusters,2),dtype='int')

    height = im.shape[0]
    width = im.shape[1]

    xv, yv = np.meshgrid(
        np.linspace(0 + height/(np.sqrt(num_clusters) + 1), height - height/(np.sqrt(num_clusters) + 1),
                    np.int16(np.sqrt(num_clusters))),
        np.linspace(0 + width/(np.sqrt(num_clusters) + 1), width - width/(np.sqrt(num_clusters) + 1),
                    np.int16(np.sqrt(num_clusters))))

    xv.shape = [num_clusters]
    yv.shape = [num_clusters]

    cluster_centers[:, 0] = xv
    cluster_centers[:, 1] = yv

    im_gradient = get_gradients(im)

    for i in range(num_clusters):

        center_point = cluster_centers[i, :]

        searching_area = im_gradient[center_point[0]-1:center_point[0]+2, center_point[1]-1:center_point[1]+2]

        mini_index = np.unravel_index(np.argmin(searching_area, axis=None), searching_area.shape)

        cluster_centers[i, :] = mini_index + center_point - [1, 1]

    return cluster_centers

def slic(im,num_clusters,cluster_centers):
    # Implement the slic function such that all pixels assigned to a label
    # should be close to each other in squared distance of augmented vectors.
    # You can weight the color and spatial components of the augmented vectors
    # differently. To do this, experiment with different values of spatial_weight.
    h,w,c = im.shape
    S = int(np.sqrt(h*w / num_clusters))

    spatial_weight = 0.3

    im_aug = np.zeros([h, w, 5], dtype=np.float32)
    im_aug[:, :, 0] = im[:, :, 0]
    im_aug[:, :, 1] = im[:, :, 1]
    im_aug[:, :, 2] = im[:, :, 2]

    xv, yv = np.meshgrid(np.arange(0, h, 1), np.arange(0, w, 1))

    im_aug[:, :, 3] = spatial_weight * xv
    im_aug[:, :, 4] = spatial_weight * yv

    clusters = np.zeros([h, w], dtype=np.int16)
    min_dis = np.zeros([h, w], dtype=np.float32)
    min_dis[:, :] = np.inf

    cc_aug = np.zeros([num_clusters, 5], dtype=np.float32)

    for i in range(num_clusters):
        cc_aug[i, :] = im_aug[cluster_centers[i, 0], cluster_centers[i, 1], :]

    for iter_ in range(50):

        for i in range(num_clusters):

            if np.isnan(cc_aug[i, 3]) or np.isnan(cc_aug[i, 4]):
                continue

            # Set Searching Windows as 2S * 2S
            x = int(cc_aug[i, 3]/spatial_weight)
            y = int(cc_aug[i, 4]/spatial_weight)

            x_min = np.max([x - S, 0])
            x_max = np.min([x + S, h])
            y_min = np.max([y - S, 0])
            y_max = np.min([y + S, w])

            src_cmp = np.zeros([h, w])
            src_cmp[:, :] = np.inf
            src_cmp[x_min:x_max, y_min:y_max] = np.sqrt(np.sum(np.square(im_aug[x_min:x_max, y_min:y_max] -
                                                                        cc_aug[i, :]), 2))

            # Set New cluster center
            clusters[src_cmp < min_dis] = i

            # Update min_dis
            min_dis[src_cmp < min_dis] = src_cmp[src_cmp < min_dis]

        # Update new cluster centers
        cc_aug_cmp = cc_aug.copy()
        for i in range(num_clusters):

            cc_aug[i, :] = np.mean(im_aug[clusters == i], 0)

        # If iter finishs
        if np.sum(np.sqrt(np.sum(np.square(cc_aug_cmp - cc_aug), 1)), 0) < 0.5:
            print('Finish in Iter: ', iter_)
            break

        if (iter_ + 1) == 50:
            print('Finish in Iter: ', (iter_ + 1))

    return clusters

########################## Support code below

from skimage.io import imread, imsave
from os.path import normpath as fn # Fixes window/linux path conventions
import matplotlib.cm as cm
import warnings
warnings.filterwarnings('ignore')

# Use get_gradients (code from pset1) to get the gradient of your image when initializing your cluster centers.
def get_gradients(im):
    if len(im.shape) > 2:
        im = np.mean(im,axis=2)
    df = np.float32([[1,0,-1]])
    sf = np.float32([[1,2,1]])
    gx = conv2(im,sf.T,'same','symm')
    gx = conv2(gx,df,'same','symm')
    gy = conv2(im,sf,'same','symm')
    gy = conv2(gy,df.T,'same','symm')
    return np.sqrt(gx*gx+gy*gy)

# normalize_im normalizes our output to be between 0 and 1
def normalize_im(im):
    im += np.abs(np.min(im))
    im /= np.max(im)
    return im

# create an output image of our cluster centers
def create_centers_im(im,centers):
    for center in centers:
        im[center[0]-2:center[0]+2,center[1]-2:center[1]+2] = [255.,0.,255.]
    return im

im = np.float32(imread(fn('inputs/lion.jpg')))

num_clusters = [25,49,64,81,100]
for num_clusters in num_clusters:
    cluster_centers = get_cluster_centers(im,num_clusters)
    imsave(fn('outputs/prob1a_' + str(num_clusters)+'_centers.jpg'),normalize_im(create_centers_im(im.copy(),cluster_centers)))
    out_im = slic(im,num_clusters,cluster_centers)

    Lr = np.random.permutation(num_clusters)
    out_im = Lr[np.int32(out_im)]
    dimg = cm.jet(np.minimum(1,np.float32(out_im.flatten())/float(num_clusters)))[:,0:3]
    dimg = dimg.reshape([out_im.shape[0],out_im.shape[1],3])
    imsave(fn('outputs/prob1b_'+str(num_clusters)+'.jpg'),normalize_im(dimg))
