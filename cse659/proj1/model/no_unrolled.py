import tensorflow as tf
from model.tfbase import TFBase
import numpy as np


# noinspection PyAttributeOutsideInit
class Net(TFBase):
    def __init__(self, input_shape=(64, 64), output_shape=(64, 64), kernel=None):
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.kernel = kernel

        super().__init__((None, input_shape[0], input_shape[1], 1), (None, output_shape[0], output_shape[1], 1))

    def get_metrics(self):
        psnr = tf.image.psnr(self.y_output, self.y_gt, 1)
        ssim = tf.image.ssim(self.y_output, self.y_gt, 1)

        metrics = [self.loss, psnr, ssim]
        metrics_name = ["LOSS", "PSNR", "SSIM"]

        return [metrics, metrics_name]

    def get_train_op(self):
        return tf.train.AdamOptimizer(0.01).minimize(self.loss)

    def get_loss(self):
        return tf.losses.mean_squared_error(self.y_output, self.y_gt)

    def get_net_output(self):

        def residual_block(input_block):
            result_block = tf.layers.conv2d(input_block, 16, 3, padding='same', activation=tf.nn.relu)
            for block_conv_times in range(3):
                result_block = tf.layers.conv2d(result_block, 16, 3, padding='same', activation=tf.nn.relu)

            result_block = tf.layers.conv2d(result_block, 8, 3, padding='same', activation=tf.nn.relu)
            result_block = tf.layers.conv2d(result_block, 1, 3, padding='same', activation=None)
            result_block += input_block

            return result_block

        iter_ = 3  # To use iter times structure
        result = residual_block(self.x)
        for i in range(iter_ - 1):
            result = residual_block(result)

        return result

    @staticmethod
    def kernel_pad(k, size):
        result = np.zeros(size, dtype=np.float32)

        k_center = int((k.shape[0] - 1) / 2)
        k_up_left = k[:k_center, :k_center]
        k_down_right = k[k_center:, k_center:]

        k_up_right = k[:k_center, k_center:]
        k_down_left = k[k_center:, :k_center]

        result[:k_down_right.shape[0], :k_down_right.shape[1]] = k_down_right
        result[-1 * k_up_left.shape[0]:, -1 * k_up_left.shape[1]:] = k_up_left
        result[-1 * k_up_right.shape[0]:, : k_up_right.shape[1]] = k_up_right
        result[:k_down_left.shape[0], -1 * k_down_left.shape[1]:] = k_down_left

        return result
