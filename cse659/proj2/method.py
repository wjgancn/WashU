from tensorflow.python.keras.layers import Conv2D, BatchNormalization, ReLU, Conv2DTranspose, Input, Add, LeakyReLU
from tensorflow.python.keras import Model
import tensorflow as tf


def general_conv2d(input_, filters, kernel_size, stride, is_norm=False):
    output_ = Conv2D(filters=filters, kernel_size=kernel_size, strides=stride, padding='same')(input_)
    if is_norm:
        output_ = BatchNormalization()(output_)
    output_ = LeakyReLU()(output_)

    return output_


def general_conv2d_transpose(input_, filters, kernel_size, stride):
    output_ = Conv2DTranspose(filters=filters, kernel_size=kernel_size, strides=stride, padding='same')(input_)
    output_ = BatchNormalization()(output_)
    output_ = LeakyReLU()(output_)

    return output_


def generator(input_shape):
    _, _, input_channel = input_shape

    inputs = Input(shape=input_shape)
    outputs = Conv2D(filters=64, kernel_size=7, strides=(1, 1), padding='same')(inputs)
    outputs = general_conv2d(input_=outputs, filters=128, kernel_size=3, stride=(2, 2))
    outputs = general_conv2d(input_=outputs, filters=256, kernel_size=3, stride=(2, 2))

    outputs = Add()([general_conv2d(input_=outputs, filters=256, kernel_size=3, stride=(1, 1)), outputs])
    outputs = Add()([general_conv2d(input_=outputs, filters=256, kernel_size=3, stride=(1, 1)), outputs])
    outputs = Add()([general_conv2d(input_=outputs, filters=256, kernel_size=3, stride=(1, 1)), outputs])
    outputs = Add()([general_conv2d(input_=outputs, filters=256, kernel_size=3, stride=(1, 1)), outputs])
    outputs = Add()([general_conv2d(input_=outputs, filters=256, kernel_size=3, stride=(1, 1)), outputs])
    outputs = Add()([general_conv2d(input_=outputs, filters=256, kernel_size=3, stride=(1, 1)), outputs])

    outputs = general_conv2d_transpose(input_=outputs, filters=128, kernel_size=3, stride=(2, 2))
    outputs = general_conv2d_transpose(input_=outputs, filters=64, kernel_size=3, stride=(2, 2))

    outputs = Conv2D(filters=input_channel, kernel_size=7, strides=(1, 1), padding='same', activation='tanh')(outputs)

    return Model(inputs=inputs, outputs=outputs)


def discriminator(input_shape):
    inputs = Input(shape=input_shape)

    outputs = general_conv2d(input_=inputs, filters=64, kernel_size=4, stride=(2, 2))
    outputs = general_conv2d(input_=outputs, filters=128, kernel_size=4, stride=(2, 2))
    outputs = general_conv2d(input_=outputs, filters=256, kernel_size=4, stride=(2, 2))
    outputs = general_conv2d(input_=outputs, filters=512, kernel_size=4, stride=(2, 2))
    outputs = general_conv2d(input_=outputs, filters=1, kernel_size=1, stride=(1, 1))
    return Model(inputs=inputs, outputs=outputs)


class CallBackFun:
    def __init__(self, output_path, train_dataset: tf.data.Dataset, valid_dataset: tf.data.Dataset):
        self.writer = tf.contrib.summary.create_file_writer(output_path)
        self.global_batch = 0

        self.train_x, self.train_y = train_dataset.make_one_shot_iterator().get_next()
        self.valid_x, self.valid_y = valid_dataset.make_one_shot_iterator().get_next()

    def on_train_begin(self):
        with self.writer.as_default(), tf.contrib.summary.always_record_summaries():
            tf.contrib.summary.image(name='train/x', tensor=self.train_x, step=0, max_images=5)
            tf.contrib.summary.image(name='train/y', tensor=self.train_y, step=0, max_images=5)

            tf.contrib.summary.image(name='valid/x', tensor=self.valid_x, step=0, max_images=5)
            tf.contrib.summary.image(name='valid/y', tensor=self.valid_y, step=0, max_images=5)

    def on_train_batch_end(self, logs=None):
        with self.writer.as_default(), tf.contrib.summary.always_record_summaries():
            tf.contrib.summary.scalar(name='train_batch/loss_gen', tensor=logs['loss_gen'], step=self.global_batch)
            tf.contrib.summary.scalar(name='train_batch/loss_dis', tensor=logs['loss_dis'], step=self.global_batch)

        self.global_batch += 1

    def on_epoch_end(self, epoch, gen_xy: tf.keras.models.Model, gen_yx: tf.keras.models.Model = None, logs=None):
        train_predict = gen_xy(self.train_x, training=False)
        train_recon = gen_yx(train_predict, training=False)

        valid_predict = gen_xy(self.valid_x, training=False)
        valid_recon = gen_yx(valid_predict, training=False)

        with self.writer.as_default(), tf.contrib.summary.always_record_summaries():

            tf.contrib.summary.image(name='train/predict', tensor=train_predict, step=epoch, max_images=5)
            tf.contrib.summary.image(name='train/recon', tensor=train_recon, step=epoch, max_images=5)

            tf.contrib.summary.image(name='valid/predict', tensor=valid_predict, step=epoch, max_images=5)
            tf.contrib.summary.image(name='valid/recon', tensor=valid_recon, step=epoch, max_images=5)
