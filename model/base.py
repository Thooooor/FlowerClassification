# -*- encoding: utf-8 -*-
"""
@Project    :   FlowerClassification
@File       :   base.py
@Time       :   2021/4/28 22:30
@Author     :   Thooooor
@Version    :   1.0
@Contact    :   thooooor999@gmail.com
@Describe   :   None
"""
from datetime import datetime

import numpy as np
import pandas as pd
import os
import abc

from dataset import Dataset
from preprocess import Processor
from utils.utils import load_model
from pprint import pprint
from utils.plot import plot_confusion_matrix, plot_stacked_bar


class Model(metaclass=abc.ABCMeta):
    """
    Base Model for later models
    """
    def __init__(self, dataset: Dataset, model_name):
        self.dataset = dataset
        self.processor = Processor()
        self.model_name = model_name
        self.model_path = "saved_model/%s.pickle" % model_name
        if os.path.exists(self.model_path):
            self.model = load_model(self.model_path)
            print(str(datetime.now()) + "\tLoaded Model From %s." % self.model_path)
            pprint(self.model.best_params_)
        else:
            self.model = None

    def train(self):
        """
        train model on train set
        :return:
        """
        start = datetime.now()
        print(str(start) + "\tStarted training model_name: %s" % self.model_name)
        X_train = self.processor.process(self.dataset.X_train)
        y_train = self.dataset.y_train
        self.model.fit(X_train, y_train)
        end = datetime.now()
        print(str(end) + "\tFinished model_name training. Cost Time: " + str(end - start))

    def test(self):
        """
        run model on valid set
        :return: accuracy score
        """
        start = datetime.now()
        print(str(start) + "\tRunning on Valid Set")
        X_test = self.processor.process(self.dataset.X_valid)
        y_test = self.dataset.y_valid
        y_pred = self.model.predict(X_test)
        score = 100 * np.sum(y_pred == y_test) / len(y_test)
        end = datetime.now()
        print(str(end) + "\tModel:%s\tPercentage Correct: %.4f" %
              (self.model_name, score))
        plot_confusion_matrix(y_pred, y_test, self.dataset.targets, self.model_name)
        plot_stacked_bar(y_pred, y_test, self.dataset.targets, self.model_name)
        return score

    def predict(self):
        """
        run model on test set, results saved to result.csv
        :return:
        """
        X_pred = self.processor.process(self.dataset.X_test)
        y_pred = self.model.predict(X_pred)
        self.write_csv(y_pred)

    def write_csv(self, results):
        """
        write results to csv file
        :param results:
        :return:
        """
        file_path = "result.csv"
        result_df = pd.DataFrame(data={"id": self.dataset.test_id, "result": results})
        result_df.to_csv(file_path, index=False, header=False)
        print(str(datetime.now()) + "\tPredict Results saved to '%s'" % file_path)

    @abc.abstractmethod
    def grid_search(self):
        """
        grid search for best parameter on train set
        :return:
        """
        pass
