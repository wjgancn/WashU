import numpy as np
from PIL import Image
from PyQt5 import QtGui
import cv2 as cv
from PIL import ImageEnhance


class TiffVideoLoader:

    def __init__(self, file_path):

        self.file_path = file_path

        tiffrawdata = Image.open(self.file_path)

        self.tiffarray_size = tiffrawdata.size
        self.tiffarray_maxframe = tiffrawdata.n_frames

        self.tiffarray = np.zeros([tiffrawdata.n_frames, self.tiffarray_size[1], self.tiffarray_size[0]])
        self.tiffarray_ad = np.zeros([tiffrawdata.n_frames, self.tiffarray_size[1], self.tiffarray_size[0]])

        for i in range(self.tiffarray_maxframe):
            tiffrawdata.seek(i)
            self.tiffarray[i, :, :] = np.array(tiffrawdata)
            # 0 - 1 Normalization
            self.tiffarray[i, :, :] = 255 * ((self.tiffarray[i, :, :] - np.amin(self.tiffarray[i, :, :])) \
                                      / np.amax(self.tiffarray[i, :, :]))

            im = cv.GaussianBlur(self.tiffarray[i, :, :], (7, 7), 10, 10)
            im = Image.fromarray(np.uint8(im))
            enh_con = ImageEnhance.Contrast(im)
            contrast = 3.5
            im = enh_con.enhance(contrast)
            im = np.asarray(im)
            self.tiffarray_ad[i, :, :] = im

        self.tiffarray = self.tiffarray.astype(np.uint8)
        self.tiffarray_ad = self.tiffarray_ad.astype(np.uint8)

    def read_frame_nparray(self, frame_index):

        if frame_index >= self.tiffarray_maxframe or frame_index < 0:
            return None
        else:
            return self.tiffarray_ad[frame_index, :, :]

    def read_frame_nparray_original(self, frame_index):

        if frame_index >= self.tiffarray_maxframe or frame_index < 0:
            return None
        else:
            return self.tiffarray[frame_index, :, :]