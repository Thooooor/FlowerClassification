# -*- encoding: utf-8 -*-
"""
@Project    :   FlowerClassification
@File       :   set_type.py
@Time       :   2021/4/15 17:38
@Author     :   Thooooor
@Version    :   1.0
@Contact    :   thooooor999@gmail.com
@Describe   :   None
"""
import os

import cv2 as cv
import numpy as np
from tqdm import tqdm
from datetime import datetime

height = 500
width = 500


class Dataset:
    """
    class for dataset,
    load all data including training, valid and test data.
    """
    def __init__(self):
        self.targets = ["passion", "watercress", "waterlily"]

        self.X_train = []
        self.y_train = []
        self.X_valid = []
        self.y_valid = []
        self.X_test = []
        self.test_id = []

        self.passion_train_set = []
        self.watercress_train_set = []
        self.waterlily_train_set = []
        self.passion_valid_set = []
        self.watercress_valid_set = []
        self.waterlily_valid_set = []
        self.test_set = []

        self.train_file_path = "dataset/train/"
        self.valid_file_path = "dataset/validation/"
        self.test_file_path = "dataset/test/"

        self.load_all_data()
        print(str(datetime.now()) + "\tLoaded All Data.")

    def load_all_data(self):
        self.passion_train_set, self.watercress_train_set, self.waterlily_train_set, self.X_train, self.y_train = \
            self.load_all_images(self.train_file_path)

        self.passion_valid_set, self.watercress_valid_set, self.waterlily_valid_set, self.X_valid, self.y_valid = \
            self.load_all_images(self.valid_file_path)

        self.load_test_images()

    @staticmethod
    def load_all_images(file_path):
        passion_set = []
        watercress_set = []
        waterlily_set = []
        X = []
        y = []
        passion_path = file_path + "passion/"
        watercress_path = file_path + "watercress/"
        waterlily_path = file_path + "waterlily/"

        passion_files = os.listdir(passion_path)
        for passion_file in tqdm(passion_files):
            img = cv.imread(passion_path + passion_file)
            passion_set.append(cv.resize(img, (width, height)))

        watercress_files = os.listdir(watercress_path)
        for watercress_file in tqdm(watercress_files):
            img = cv.imread(watercress_path + watercress_file)
            watercress_set.append(cv.resize(img, (width, height)))

        waterlily_files = os.listdir(waterlily_path)
        for waterlily_file in tqdm(waterlily_files):
            img = cv.imread(waterlily_path + waterlily_file)
            waterlily_set.append(cv.resize(img, (width, height)))

        for passion in passion_set:
            X.append(np.array(passion))
            y.append("passion")

        for watercress in watercress_set:
            X.append(np.array(watercress))
            y.append("watercress")

        for waterlily in waterlily_set:
            X.append(np.array(waterlily))
            y.append("waterlily")

        return passion_set, watercress_set, waterlily_set, X, y

    def load_test_images(self):
        test_files = os.listdir(self.test_file_path)
        for test_file in tqdm(test_files):
            img = cv.imread(self.test_file_path + test_file)
            self.test_set.append(cv.resize(img, (width, height)))
            self.test_id.append(test_file.split('.')[0])
        for data in self.test_set:
            self.X_test.append(np.array(data))


if __name__ == '__main__':
    ds = Dataset()
    cv.imshow("test", ds.waterlily_valid_set[0])
    cv.waitKey(0)
