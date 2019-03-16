import tensorflow as tf
from model.tfbase import TFBase
import numpy as np
from scipy.signal import convolve2d as conv2d
import os


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
        return tf.train.AdamOptimizer(0.001).minimize(self.loss)

    def get_loss(self):
        return tf.losses.mean_squared_error(self.y_output, self.y_gt)

    def get_net_output(self):
        result = tf.layers.conv2d(self.x, 16, 3, padding='same', activation=tf.nn.relu)
        for block_conv_times in range(5):
            result = tf.layers.conv2d(result, 16, 3, padding='same', activation=tf.nn.relu)

        result = tf.layers.conv2d(result, 8, 3, padding='same', activation=tf.nn.relu)
        result = tf.layers.conv2d(result, 1, 3, padding='same', activation=None)

        return result

    def solve(self, x, iter_max, tol_max, model_path):

        pre = np.zeros(shape=x.shape, dtype=np.float64)

        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
        for i in range(x.shape[0]):
            x_batch = np.squeeze(x[i], -1)

            kt_y = conv2d(x_batch, self.kernel.transpose(), mode='same')

            u = 0.5

            from model.unrolled import Net

            fre_k = Net.kernel_pad(self.kernel, self.input_shape)
            fre_k = np.fft.fft2(fre_k)
            fre_k = 2*u + np.square(np.abs(fre_k))

            result = x_batch

            for iter_ in range(iter_max):

                result_last = result

                result = kt_y + 2*u*result
                result = np.fft.fft2(result)
                result = result / fre_k
                result = np.fft.ifft2(result)
                result = np.abs(result)

                result = np.expand_dims(result, 0)
                result = np.expand_dims(result, -1)
                result = self.predict(result, 1, model_path=model_path)

                result = np.squeeze(result, 0)
                result = np.squeeze(result, -1)

                tol_ = np.mean(np.square(result - result_last))
                print(tol_)
                if tol_ < tol_max:
                    verbose_info = "[Info] Prediction Output: Break in [%d] Iter" % (iter_ + 1)
                    print(verbose_info)
                    break

            result = np.expand_dims(result, -1)
            pre[i] = result

            verbose_info = "[Info] Prediction Output: Batch = [%d]" % (i + 1)
            print(verbose_info)

        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'
        return pre
