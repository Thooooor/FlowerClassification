# -*- encoding: utf-8 -*-
"""
@Project    :   FlowerClassification
@File       :   voting.py
@Time       :   2021/4/29 1:15
@Author     :   Thooooor
@Version    :   1.0
@Contact    :   thooooor999@gmail.com
@Describe   :   None
"""
from datetime import datetime
from pprint import pprint

from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC

from dataset import Dataset
from model.base import Model
from utils.utils import save_model


class Voting(Model):
    """
    Voting model for image classifier, combining SVM and SGD
    """
    def __init__(self, dataset: Dataset):
        super(Voting, self).__init__(dataset, "Voting Classifier")
        if self.model is None:
            self.model = self.grid_search()

    def grid_search(self):
        param_grid = {
            'voting': ['hard', 'soft'],
            'weights': [None, (6, 4, 0), (7, 3, 0), (8, 2, 0), (4, 6, 0), (7, 2, 1), (8, 1, 1), (6, 2, 2), (6, 3, 1)]
        }
        start = datetime.now()
        print(str(start) + "\tStarted grid search for %s" % self.model_name)
        pprint(param_grid)
        clf1 = SVC(kernel='linear', probability=True)
        clf2 = SGDClassifier(loss='log')
        clf3 = KNeighborsClassifier(weights='distance')
        model = VotingClassifier(estimators=[
            ('svc', clf1), ('sgd', clf2), ('knn', clf3)
        ])
        grid_search = GridSearchCV(
            model,
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
