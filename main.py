# -*- encoding: utf-8 -*-
"""
@Project    :   FlowerClassification
@File       :   main.py
@Time       :   2021/4/20 15:31
@Author     :   Thooooor
@Version    :   1.0
@Contact    :   thooooor999@gmail.com
@Describe   :   None
"""
from model.knn import KNN
from model.svm import SVM
from model.sgd import SGD
from model.rf import RF
from model.voting import Voting
from dataset import Dataset

dataset = Dataset()


def svm():
    model = SVM(dataset)
    model.test()


def sgd():
    model = SGD(dataset)
    model.test()


def knn():
    model = KNN(dataset)
    model.test()


def rf():
    model = RF(dataset)
    model.test()


def voting():
    model = Voting(dataset)
    model.test()
    model.predict()


if __name__ == '__main__':
    voting()
