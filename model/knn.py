# -*- encoding: utf-8 -*-
"""
@Project    :   FlowerClassification
@File       :   knn.py
@Time       :   2021/4/19 21:29
@Author     :   Thooooor
@Version    :   1.0
@Contact    :   thooooor999@gmail.com
@Describe   :   None
"""
from datetime import datetime
from pprint import pprint

from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier

from dataset import Dataset
from model.base import Model
from utils.utils import save_model


class KNN(Model):
    """
    KNN model for image classifier
    """

    def __init__(self, dataset: Dataset):
        super(KNN, self).__init__(dataset, "KNN")
        if self.model is None:
            self.model = self.grid_search()

    def grid_search(self):
        param_grid = {
            'weights': ['uniform', 'distance'],
            'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
        }
        start = datetime.now()
        print(str(start) + "\tStarted grid search for %s" % self.model_name)
        pprint(param_grid)

        grid_search = GridSearchCV(
            KNeighborsClassifier(),
            param_grid,
            cv=5,
            n_jobs=-1,
            scoring='accuracy',
            verbose=1,
            return_train_score=True
        )
        X_train = self.processor.process(self.dataset.X_train)
        y_train = self.dataset.y_train
        grid_result = grid_search.fit(X_train, y_train)

        end = datetime.now()
        print(str(end) + "\tFinished grid search for %s. Cost: %s" % (self.model, str(end - start)))
        print("Best Score: %.4f" % grid_result.best_score_)
        pprint(grid_result.best_params_)
        save_model(grid_result, self.model_path)
        print("Model saved to %s" % self.model_path)
        return grid_result
