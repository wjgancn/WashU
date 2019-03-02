import tensorflow as tf


class Net(object):
    def __init__(self):
        self.x = tf.placeholder(dtype=tf.float64, shape=(None, None, None, 1))
        self.y_gt = tf.placeholder(dtype=tf.float64, shape=(None, None, None, 1))

        self.y_output = self.build_network(self.x)
        self.loss = tf.losses.mean_squared_error(self.y_output, self.y_gt)
        self.train_op = tf.train.AdamOptimizer(0.01).minimize(self.loss)
        self.psnr = tf.image.psnr(self.y_output, self.y_gt, 1)
        self.ssim = tf.image.ssim(self.y_output, self.y_gt, 1)

        self.metrics = [self.loss, self.psnr, self.ssim]
        self.metrics_name = ["Loss", "PSNR", "SSIM"]

    @staticmethod
    def build_network(input_):
        result = tf.layers.conv2d(input_, 64, 3, padding='same', activation=tf.nn.relu)
        result = tf.layers.conv2d(result, 32, 3, padding='same', activation=tf.nn.relu)
        result = tf.layers.conv2d(result, 16, 3, padding='same', activation=tf.nn.relu)
        result = tf.layers.conv2d(result, 1, 3, padding='same', activation=tf.nn.relu)

        return result
