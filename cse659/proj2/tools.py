import shutil
import logging
import os
import tensorflow as tf
import numpy as np


def psnr_tf(y_true, y_pred):
    return tf.image.psnr(y_true, y_pred, max_val=1)


def ssim_tf(y_true, y_pred):
    return tf.image.ssim(y_true, y_pred, max_val=1)


def copy_code(src, dst):
    for i in range(100):  # 100 is picked manually : )
        path = dst + 'code_' + str(i) + '/'
        if not os.path.exists(path):
            shutil.copytree(src=src, dst=path)  # Copy and back current codes.
            return True


def set_logging_file(path):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    log_file = logging.FileHandler(filename=path + 'log.txt')
    log_file.setFormatter(logging.Formatter("%(asctime)s %(name)s %(levelname)s %(message)s"))
    logging.root.addHandler(log_file)
    return True


def save_predict(imgs, evaluation, name, path, slim=False):
    import PIL
    import scipy.io as sio

    batches, depths, width, height, channel = imgs.shape
    if slim:
        depths = 1

    imgs_list = []
    for depth in range(depths):
        for batch in range(batches):
            img_current = np.squeeze(imgs[batch, depth, :, :, :])
            img_current -= np.amin(img_current)
            img_current /= np.amax(img_current)

            imgs_list.append(PIL.Image.fromarray(img_current))

    imgs_list[0].save(path + name + '.tiff', save_all=True, append_images=imgs_list[1:])

    f = open(path + name + '.txt', 'w')
    f.writelines(str(evaluation))
    f.close()

    sio.savemat(path + name + '.mat', {'predict': imgs})
