import tensorflow as tf


class Net(object):
    def __init__(self, mode):
        self.x = tf.placeholder(dtype=tf.float64, shape=(None, None, None, 1))
        self.y_gt = tf.placeholder(dtype=tf.float64, shape=(None, None, None, 1))
        self.y_output = self.build_network(self.x, mode)

        self.loss = tf.losses.mean_squared_error(self.y_output, self.y_gt)
        self.psnr = tf.image.psnr(self.y_output, self.y_gt, 1)
        self.ssim = tf.image.ssim(self.y_output, self.y_gt, 1)

        self.metrics = [self.loss, self.psnr, self.ssim]
        self.metrics_name = ["Loss", "PSNR", "SSIM"]

        self.train_op = tf.train.AdamOptimizer(0.01).minimize(self.loss)

    @staticmethod
    def build_network(input_, mode):
        def no_unrolled():
            def residual_block(input_block):
                result_block = tf.layers.conv2d(input_block, 32, 3, padding='same', activation=tf.nn.relu)
                for block_conv_times in range(3):
                    result_block = tf.layers.conv2d(result_block, 32, 3, padding='same', activation=tf.nn.relu)

                result_block = tf.layers.conv2d(result_block, 16, 3, padding='same', activation=tf.nn.relu)
                result_block = tf.layers.conv2d(result_block, 1, 3, padding='same', activation=None)
                result_block += input_block

                return result_block

            result = residual_block(input_)
            result = residual_block(result)
            result = residual_block(result)
            result = residual_block(result)
            result = residual_block(result)

            return result

        mode_dict = {
            "no_unrolled": no_unrolled
        }

        return mode_dict[mode]()
