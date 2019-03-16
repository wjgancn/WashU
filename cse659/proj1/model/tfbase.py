import tensorflow as tf
import numpy as np
import os
import shutil
from tensorboardX import SummaryWriter
import scipy.io as sio


# noinspection PyMethodMayBeStatic
class TFBase(object):
    def __init__(self, input_shape, output_shape):
        self.x = tf.placeholder(dtype=tf.float64, shape=input_shape)
        self.y_gt = tf.placeholder(dtype=tf.float64, shape=output_shape)
        self.y_output = self.get_net_output()

        self.loss = self.get_loss()
        self.train_op = self.get_train_op()

        self.metrics, self.metrics_name = self.get_metrics()

    def get_net_output(self):
        return 0

    def get_metrics(self):
        return [0, 0]

    def get_train_op(self):
        return 0

    def get_loss(self):
        return 0


class TFTrainer(object):
    def __init__(self, net, path, config_info, batch_size=32, train_epoch=100, save_epoch=20):
        """

        :type net: Should have 'x', 'y_gt', 'y_output', 'matrices', 'matrices_name' and 'train_op' object members
        """
        self.save_epoch = save_epoch
        self.train_epoch = train_epoch
        self.batch_size = batch_size

        self.net = net
        self.path = path
        new_folder(path)
        self.config_info = config_info

    def run(self, train_x, train_y, valid_x, valid_y, train_imgs_index=np.array([0]), valid_imgs_index=np.array([0]),
            batch_verbose=True, epoch_verbose=True):

        train_x_imgs = train_x[train_imgs_index]
        train_y_imgs = train_y[train_imgs_index]
        valid_x_imgs = valid_x[valid_imgs_index]
        valid_y_imgs = valid_y[valid_imgs_index]

        ################
        # Set up shuffle index
        ################
        nums_train = train_x.shape[0]
        nums_valid = valid_x.shape[0]

        index_batches_train = self.make_batches(nums_train, self.batch_size)
        index_train = np.arange(nums_train)

        index_batches_valid = self.make_batches(nums_valid, self.batch_size)

        num_batches_train = index_batches_train.__len__()
        num_batches_valid = index_batches_valid.__len__()

        nums_metrics = self.net.metrics_name.__len__()

        batch_verbose_times = None
        if batch_verbose:
            batch_verbose_times = int(num_batches_train / 10)

        ################
        # Initial writer
        ################
        writer = TBXWriter(self.path, self.config_info)
        writer.imgs_train_init(train_x_imgs, train_y_imgs)
        writer.imgs_valid_init(valid_x_imgs, valid_y_imgs)

        ################
        # Initial .mat save path
        ################
        imgs_save_path = self.path + "mat/"
        new_folder(imgs_save_path)
        sio.savemat(imgs_save_path + "init.mat", {"x": valid_x_imgs, "y": valid_y_imgs})

        gpu_options = tf.GPUOptions(allow_growth=True)
        with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
            sess.run(tf.global_variables_initializer())

            writer_step = 0
            for epoch in range(self.train_epoch):

                ################
                # Training
                ################
                np.random.shuffle(index_train)
                epoch_metrics = np.zeros(shape=[nums_metrics, 1])
                for batch, (batch_start, batch_end) in enumerate(index_batches_train):
                    batch_ids = index_train[batch_start:batch_end]
                    x = train_x[batch_ids]
                    y_gt = train_y[batch_ids]

                    ################
                    # Run
                    ################
                    _, metrics = sess.run([self.net.train_op, self.net.metrics],
                                          feed_dict={self.net.x: x, self.net.y_gt: y_gt})

                    ################
                    # Verbose
                    ################
                    verbose_info = "[Info] Batch Output: Batch = [%d] Totally [%d]. " \
                                   % (batch + 1, num_batches_train)

                    for i in range(nums_metrics):
                        metric = metrics[i]
                        metric = metric.mean()

                        epoch_metrics[i] += metric
                        verbose_info += self.net.metrics_name[i] + ": [%.4f]. " % metric

                    if batch_verbose:
                        if (batch + 1) % batch_verbose_times == 0:
                            print(verbose_info)

                    ################
                    # Add data into writer
                    ################
                    writer.train_batch(metrics, self.net.metrics_name, writer_step)
                    writer_step += 1

                verbose_info = "[Info] Epoch Output: Epoch = [%d] Totally [%d]. " % (epoch + 1, self.train_epoch)
                for i in range(nums_metrics):
                    epoch_metrics[i] /= num_batches_train
                    verbose_info += self.net.metrics_name[i] + ": [%.4f]. " % epoch_metrics[i]

                if epoch_verbose:
                    print(verbose_info)

                ################
                # Add data into writer
                ################
                writer.train_epoch(epoch_metrics, self.net.metrics_name, writer_step)
                epoch_imgs = sess.run(self.net.y_output, feed_dict={self.net.x: train_x_imgs})
                writer.imgs_train_epoch(epoch_imgs, writer_step)

                ################
                # Validation
                ################
                epoch_metrics = np.zeros(shape=[nums_metrics, 1])
                for batch, (batch_start, batch_end) in enumerate(index_batches_valid):
                    x = valid_x[batch_start: batch_end]
                    y_gt = valid_y[batch_start: batch_end]

                    metrics = sess.run(self.net.metrics, feed_dict={self.net.x: x, self.net.y_gt: y_gt})

                    for i in range(nums_metrics):
                        metric = metrics[i]
                        metric = metric.mean()
                        epoch_metrics[i] += metric

                ################
                # Verbose
                ################
                verbose_info = "[Info] Validation Output: Epoch = [%d] Totally [%d]. " % (epoch + 1, self.train_epoch)
                for i in range(nums_metrics):
                    epoch_metrics[i] /= num_batches_valid
                    verbose_info += self.net.metrics_name[i] + ": [%.4f]. " % epoch_metrics[i]

                if epoch_verbose:
                    print(verbose_info)

                ################
                # Add data into writer
                ################
                writer.valid_epoch(epoch_metrics, self.net.metrics_name, writer_step)
                epoch_imgs = sess.run(self.net.y_output, feed_dict={self.net.x: valid_x_imgs})
                writer.imgs_valid_epoch(epoch_imgs, writer_step)
                ################
                # Save .mat
                ################
                sio.savemat(imgs_save_path + "epoch_%d.mat" % (epoch + 1), {"pre": epoch_imgs})

                ################
                # Save Model
                ################
                if (epoch + 1) % self.save_epoch == 0:
                    model_save_path = self.path + "model/epoch_" + str(epoch + 1) + "/"
                    new_folder(model_save_path)
                    self.save_model(sess, model_save_path)

    @staticmethod
    def make_batches(size, batch_size):
        num_batches = (size + batch_size - 1) // batch_size  # round up
        return [(i * batch_size, min(size, (i + 1) * batch_size))
                for i in range(num_batches)]

    @staticmethod
    def save_model(sess, path):
        saver = tf.train.Saver()
        saver.save(sess, save_path=path + 'model.ckpt')

    @staticmethod
    def restore_model(sess, path):
        saver = tf.train.Saver()
        saver.restore(sess, save_path=path + 'model.ckpt')


# Clean output folder
def new_folder(path):
    if os.path.exists(path):
        shutil.rmtree(path, ignore_errors=True)
    os.makedirs(path)


# Convert a python dict to markdown table
def config_to_markdown_table(config_dict, name_section):
    info = '## ' + name_section + '\n'
    info = info + '|  Key  |  Value |\n|:----:|:---:|\n'

    for i in config_dict.keys():
        info = info + '|' + i + '|' + config_dict[i] + '|\n'

    info = info + '\n'

    return info


# A custom TensorboardX Class
class TBXWriter(object):
    def __init__(self, path, config_info):
        self.writer = SummaryWriter(path)
        self.writer.add_text(tag='config', text_string=config_info, global_step=0)

    ###############
    # Metrics
    ###############

    def train_batch(self, metrics, metrics_name, step):
        for i in range(metrics_name.__len__()):
            self.writer.add_scalar(tag='train_batch/' + metrics_name[i], scalar_value=metrics[i].mean(), global_step=step)

    def train_epoch(self, metrics, metrics_name, step):
        for i in range(metrics_name.__len__()):
            self.writer.add_scalar(tag='train_epoch/' + metrics_name[i], scalar_value=metrics[i].mean(), global_step=step)

    def valid_epoch(self, metrics, metrics_name, step):
        for i in range(metrics_name.__len__()):
            self.writer.add_scalar(tag='valid_epoch/' + metrics_name[i], scalar_value=metrics[i].mean(), global_step=step)

    ###############
    # Image
    ###############
    @staticmethod
    def img_preprocess(imgs):
        imgs -= np.amin(imgs)
        imgs /= np.amax(imgs)
        return imgs

    def imgs_train_epoch(self, imgs, step):
        width_img = imgs.shape[1]
        height_img = imgs.shape[2]
        channel = imgs.shape[3]

        for i in range(imgs.shape[0]):
            for j in range(channel):
                self.writer.add_image(tag='train/index%d_pre_channel%d' % (i, j),
                                      img_tensor=self.img_preprocess(imgs[i].reshape([width_img, height_img])),
                                      global_step=step, dataformats='HW')

    def imgs_train_init(self, x_imgs, y_imgs):
        width_img = x_imgs.shape[1]
        height_img = x_imgs.shape[2]

        channel_x = x_imgs.shape[3]
        channel_y = y_imgs.shape[3]

        for i in range(x_imgs.shape[0]):
            for j in range(channel_x):
                self.writer.add_image(tag='train/index%d_x_channel%d' % (i, j),
                                      img_tensor=self.img_preprocess(x_imgs[i, :, :, j].reshape([width_img, height_img])),
                                      global_step=0, dataformats='HW')
            for j in range(channel_y):
                self.writer.add_image(tag='train/index%d_y_channel%d' % (i, j),
                                      img_tensor=self.img_preprocess(y_imgs[i, :, :, j].reshape([width_img, height_img])),
                                      global_step=0, dataformats='HW')

    def imgs_valid_epoch(self, imgs, step):
        width_img = imgs.shape[1]
        height_img = imgs.shape[2]
        channel = imgs.shape[3]

        for i in range(imgs.shape[0]):
            for j in range(channel):
                self.writer.add_image(tag='valid/index%d_pre_channel%d' % (i, j),
                                      img_tensor=self.img_preprocess(imgs[i].reshape([width_img, height_img])),
                                      global_step=step, dataformats='HW')

    def imgs_valid_init(self, x_imgs, y_imgs):
        width_img = x_imgs.shape[1]
        height_img = x_imgs.shape[2]

        channel_x = x_imgs.shape[3]
        channel_y = y_imgs.shape[3]

        for i in range(x_imgs.shape[0]):
            for j in range(channel_x):
                self.writer.add_image(tag='valid/index%d_x_channel%d' % (i, j),
                                      img_tensor=self.img_preprocess(x_imgs[i, :, :, j].reshape([width_img, height_img])),
                                      global_step=0, dataformats='HW')
            for j in range(channel_y):
                self.writer.add_image(tag='valid/index%d_y_channel%d' % (i, j),
                                      img_tensor=self.img_preprocess(y_imgs[i, :, :, j].reshape([width_img, height_img])),
                                      global_step=0, dataformats='HW')
