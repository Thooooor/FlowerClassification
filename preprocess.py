# -*- encoding: utf-8 -*-
"""
@Project    :   FlowerClassification
@File       :   preprocess.py
@Time       :   2021/4/15 17:32
@Author     :   Thooooor
@Version    :   1.0
@Contact    :   thooooor999@gmail.com
@Describe   :   None
"""
import cv2 as cv
import numpy as np
from skimage.feature import hog
from sklearn.preprocessing import StandardScaler


class RGB2GrayTransformer:
    """
    Convert RGB image to gray
    """
    def __init__(self):
        pass

    @staticmethod
    def transform(X):
        return np.array([cv.cvtColor(img, cv.COLOR_BGR2GRAY) for img in X])


class HogTransformer:
    """
    HOG for feature reduction
    """
    def __init__(self, y=None, orientations=8,
                 pixels_per_cell=(10, 10),
                 cells_per_block=(2, 2),
                 block_norm='L2-Hys'):
        self.y = y
        self.orientations = orientations
        self.pixels_per_cell = pixels_per_cell
        self.cells_per_block = cells_per_block
        self.block_norm = block_norm

    def transform(self, X):
        def local_hog(x):
            return hog(x,
                       orientations=self.orientations,
                       pixels_per_cell=self.pixels_per_cell,
                       cells_per_block=self.cells_per_block,
                       block_norm=self.block_norm)

        try:  # parallel
            return np.array([local_hog(img) for img in X])
        except IndexError:
            return np.array([local_hog(img) for img in X])


class Processor:
    """
    apply gray transform, HOG transform and StandardScaler on image
    """
    def __init__(self, orientations=8, pixels_per_cell=(10, 10),
                 cells_per_block=(2, 2)):
        self.gray_trans = RGB2GrayTransformer()
        self.hog_trans = HogTransformer(orientations=orientations, pixels_per_cell=pixels_per_cell,
                                        cells_per_block=cells_per_block)
        self.scale_trans = StandardScaler()

    def process(self, data_set):
        data_set_gray = self.gray_trans.transform(data_set)
        data_set_hog = self.hog_trans.transform(data_set_gray)
        data_set_prepared = self.scale_trans.fit_transform(data_set_hog)
        return data_set_prepared
