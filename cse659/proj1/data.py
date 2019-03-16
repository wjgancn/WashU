from skimage.io import imread
from skimage.util import random_noise, view_as_windows
from scipy.signal import convolve2d as conv2
import numpy as np
import glob


def read_imgs(file_paths, noise_var, blur_kernel, patch_size, patch_step):
    result = []
    result_noise = []

    def patch(input_):
        result_ = view_as_windows(input_, (patch_size, patch_size),  patch_step)
        patch_nums = result_.shape[0] * result_.shape[1]
        result_ = np.ascontiguousarray(result_)
        result_.shape = [patch_nums, patch_size, patch_size]

        return result_

    file_paths = glob.glob(file_paths + "*.jpg")

    count = 0
    total_count = file_paths.__len__()
    print("[Info] Begin Reading %d Images." % total_count)
    for i in file_paths:
        img = imread(i, as_gray=True)
        img = img.astype(np.float64)

        img_noise = conv2(img, blur_kernel, mode='same')
        img_noise = random_noise(img_noise, mode='gaussian', mean=0, var=noise_var)

        img = normalization(img)
        img_noise = normalization(img_noise)

        img = patch(img)
        img_noise = patch(img_noise)

        result.append(img)
        result_noise.append(img_noise)

        count += 1
        if count % 10 == 0:
            print("[Info] Reading Images: Current %d with %d Totally." % (count, total_count))

    print("[Info] Concatenating Patches.")
    result = np.expand_dims(np.concatenate(result, 0), -1)
    result_noise = np.expand_dims(np.concatenate(result_noise, 0), -1)
    print("[Info] Output Shape: ", result.shape)

    return result_noise, result


def normalization(imgs):
    if imgs.shape.__len__() == 4:
        batch, width, height, channel = imgs.shape
        for i in range(batch):
            for j in range(channel):
                imgs[i, :, :, j] -= np.amin(imgs[i, :, :, j])
                amax = np.amax(imgs[i, :, :, j])
                imgs[i, :, :, j] /= amax

    if imgs.shape.__len__() == 3:
        width, height, channel = imgs.shape
        for j in range(channel):
            imgs[:, :, j] -= np.amin(imgs[:, :, j])
            amax = np.amax(imgs[:, :, j])
            imgs[:, :, j] /= amax

    if imgs.shape.__len__() == 2:
        imgs -= np.amin(imgs[:, :])
        imgs /= np.amax(imgs[:, :])

    return imgs
