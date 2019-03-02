import data
from model.net import Net
from model.trainer import Trainer, config_to_markdown_table
import os
import configparser

config = configparser.ConfigParser()
config.read('config.ini')
config_info = config_to_markdown_table(config._sections['GLOBAL'], 'GLOBAL')
config_info = config_info + config_to_markdown_table(config._sections['DATA'], 'DATA')
config_info = config_info + config_to_markdown_table(config._sections['NET'], 'NET')
config_info = config_info + config_to_markdown_table(config._sections['TRAIN'], 'TRAIN')

os.environ["CUDA_VISIBLE_DEVICES"] = config['GLOBAL']['gpu_index']

noised_std = float(config['DATA']['noise_std'])
gaussian_kernel = [[1 / 256, 4 / 256,  6 / 256,  4 / 256, 1 / 256],
                   [4 / 256, 16 / 256, 24 / 256, 16 / 256, 4 / 256],
                   [6 / 256, 24 / 256, 36 / 256, 24 / 256, 6 / 256],
                   [4 / 256, 16 / 256, 24 / 256, 16 / 256, 4 / 256],
                   [1 / 256, 4 / 256,  6 / 256,  4 / 256, 1 / 256]]
train_x, train_y, train_x_imgs, train_y_imgs = data.read_imgs('/export/project/gan.weijie/data/bsds500/train/', 10,
                                                              noised_std, gaussian_kernel, 48, 32)
valid_x, valid_y, valid_x_imgs, valid_y_imgs = data.read_imgs('/export/project/gan.weijie/data/bsds500/valid/', 2,
                                                              noised_std, gaussian_kernel, 48, 32)

TFNet = Net(config['NET']['mode'])
TFTrainer = Trainer(net=TFNet, config_info=config_info,
                    path=config['TRAIN']['path'],
                    batch_size=int(config['TRAIN']['batch_size']),
                    train_epoch=int(config['TRAIN']['train_epoch']),
                    save_epoch=int(config['TRAIN']['save_epoch']))

TFTrainer.run(train_x, train_y, valid_x, valid_y, train_x_imgs, train_y_imgs, valid_x_imgs, valid_y_imgs)
