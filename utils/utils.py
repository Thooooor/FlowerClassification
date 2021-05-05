# -*- encoding: utf-8 -*-
"""
@Project    :   FlowerClassification
@File       :   utils.py
@Time       :   2021/4/15 17:28
@Author     :   Thooooor
@Version    :   1.0
@Contact    :   thooooor999@gmail.com
@Describe   :   None
"""
import pickle


def save_model(model, model_path):
    """
    save model by pickle
    :param model: model after training
    :param model_path: path to store model
    :return:
    """
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)


def load_model(model_path):
    """
    load model from file
    :param model_path: model path
    :return:
    """
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    return model
