import data
from model.no_unrolled import Net as NoUnNet
from model.unrolled import Net as UnNet
from model.lrcnn import Net as LRCNNNet

from model.tfbase import TFTrainer, config_to_markdown_table
import os
import configparser
import numpy as np

config = configparser.ConfigParser()
config.read('config.ini')
config_info = config_to_markdown_table(config._sections['GLOBAL'], 'GLOBAL')
config_info = config_info + config_to_markdown_table(config._sections['DATA'], 'DATA')
config_info = config_info + config_to_markdown_table(config._sections['NET'], 'NET')
config_info = config_info + config_to_markdown_table(config._sections['TRAIN'], 'TRAIN')

os.environ["CUDA_VISIBLE_DEVICES"] = config['GLOBAL']['gpu_index']

noised_std = float(config['DATA']['noise_std'])
gaussian_kernel = np.array([[1 / 256, 4 / 256,  6 / 256,  4 / 256, 1 / 256],
                   [4 / 256, 16 / 256, 24 / 256, 16 / 256, 4 / 256],
                   [6 / 256, 24 / 256, 36 / 256, 24 / 256, 6 / 256],
                   [4 / 256, 16 / 256, 24 / 256, 16 / 256, 4 / 256],
                   [1 / 256, 4 / 256,  6 / 256,  4 / 256, 1 / 256]])

root_path = config['DATA']['root_path']
patch_size = int(config['DATA']['patch_size'])

net_dict = {
    'no unrolled': NoUnNet,
    'unrolled': UnNet,
    'lrcnn': LRCNNNet,
}

TFNet = net_dict[config['NET']['net']](input_shape=(patch_size, patch_size), output_shape=(patch_size, patch_size),
                                       kernel=gaussian_kernel)

if config['NET']['mode'] == 'train':

    if config['NET']['net'] == 'lrcnn':
        train_x, train_y = data.read_imgs(root_path + 'train/', noised_std, gaussian_kernel, patch_size, 32, is_blur=False)
        valid_x, valid_y = data.read_imgs(root_path + 'valid/', noised_std, gaussian_kernel, patch_size, 32, is_blur=False)

    else:
        train_x, train_y = data.read_imgs(root_path + 'train/', noised_std, gaussian_kernel, patch_size, 32)
        valid_x, valid_y = data.read_imgs(root_path + 'valid/', noised_std, gaussian_kernel, patch_size, 32)

    tf_trainer = TFTrainer(net=TFNet, config_info=config_info,
                           path=config['TRAIN']['path'],
                           batch_size=int(config['TRAIN']['batch_size']),
                           train_epoch=int(config['TRAIN']['train_epoch']),
                           save_epoch=int(config['TRAIN']['save_epoch']))

    train_imgs_index = np.array([100, 200])
    valid_imgs_index = np.array([100, 200])
    tf_trainer.run(train_x, train_y, valid_x, valid_y, train_imgs_index, valid_imgs_index)

if config['NET']['mode'] == 'test':

    test_x, test_y = data.read_imgs(root_path + 'test/', noised_std, gaussian_kernel, patch_size, patch_size)

    if config['NET']['net'] == 'lrcnn':
        test_pre = TFNet.solve(test_x, 100, 1e-11, model_path=config['NET']['model_path'])
    else:
        test_pre = TFNet.predict(test_x, batch_size=32, model_path=config['NET']['model_path'])

    TFNet.to_mat(test_x, 'x', config['NET']['save_path'])
    TFNet.to_mat(test_y, 'y', config['NET']['save_path'])
    TFNet.to_mat(test_pre, 'pre', config['NET']['save_path'])
