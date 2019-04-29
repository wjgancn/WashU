import tensorflow as tf
import numpy as np


class DataLoader(object):
    def __init__(self, path_x, path_y, is_pair=False):

        dataset_x, self.dataset_len, img_dataset_x = self.path_to_dataset(path_x)
        dataset_y, _, img_dataset_y = self.path_to_dataset(path_y)

        if not is_pair:
            self.tf_dataset = tf.data.Dataset.zip((dataset_x.shuffle(buffer_size=self.dataset_len), dataset_y.shuffle(buffer_size=self.dataset_len))).\
                map(self.convert_to_imgs_fn).repeat(1)
        else:
            self.tf_dataset = tf.data.Dataset.zip((dataset_x, dataset_y)).shuffle(buffer_size=self.dataset_len).\
                map(self.convert_to_imgs_fn).repeat(1)

        self.tf_imgs_dataset = tf.data.Dataset.zip((img_dataset_x, img_dataset_y)).map(self.convert_to_imgs_fn).\
            repeat(1).batch(5)

    @staticmethod
    def convert_to_imgs_fn(path_1, path_2):
        return DataLoader.read_and_process(path_1), DataLoader.read_and_process(path_2)

    @staticmethod
    def path_to_dataset(file_path):
        file_list = tf.gfile.Glob(file_path + '*.jpg')
        file_list.sort()

        dataset = tf.data.Dataset.from_tensor_slices(file_list)
        dataset_len = len(file_list)

        img_list = np.array(file_list)[np.array([10, 20, 30, 40, 50])].tolist()
        img_dataset = tf.data.Dataset.from_tensor_slices(img_list)

        return dataset, dataset_len, img_dataset

    @staticmethod
    def read_and_process(file_path):
        img = tf.read_file(file_path)
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.resize_images(img, size=(128, 128))
        img = tf.cast(img, tf.float32)
        img = img - tf.reduce_min(img)
        img = img / tf.reduce_max(img)
        img = 2 * img - 1  # -1 to 1 normalization
        return img
