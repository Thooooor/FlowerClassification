# -*- encoding: utf-8 -*-
"""
@Project    :   FlowerClassification
@File       :   plot.py
@Time       :   2021/4/29 19:28
@Author     :   Thooooor
@Version    :   1.0
@Contact    :   thooooor999@gmail.com
@Describe   :   None
"""
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix


def plot_confusion_matrix(y_pre, y_test, labels, model):
    """
    plot confusion matrix
    :param model:
    :param y_pre:
    :param y_test:
    :param labels:
    :return:
    """
    cm = confusion_matrix(y_test, y_pre)

    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.GnBu)
    plt.title('Confusion Matrix - ' + model)
    plt.colorbar()
    tick_marks = np.arange(len(labels))
    plt.xticks(tick_marks, labels)
    plt.yticks(tick_marks, labels)

    iters = np.reshape([[[i, j] for j in range(3)] for i in range(3)], (cm.size, 2))
    for i, j in iters:
        plt.text(j, i, format(cm[i, j]))

    plt.ylabel('Observed Labels')
    plt.xlabel('Predict Labels')
    plt.tight_layout()
    plt.savefig("img/%s-ConfusionMatrix.png" % model)
    plt.show()


def plot_stacked_bar(y_pre, y_test, labels, model):
    """
    plot stacked bar
    :param model:
    :param y_pre:
    :param y_test:
    :param labels:
    :return:
    """
    label1 = [0, 0, 0]
    label2 = [0, 0, 0]
    label3 = [0, 0, 0]
    for i, j in zip(y_test, y_pre):
        if i == labels[0]:
            if j == labels[0]:
                label1[0] += 1
            elif j == labels[1]:
                label1[1] += 1
            else:
                label1[2] += 1
        elif i == labels[1]:
            if j == labels[0]:
                label2[0] += 1
            elif j == labels[1]:
                label2[1] += 1
            else:
                label2[2] += 1
        else:
            if j == labels[0]:
                label3[0] += 1
            elif j == labels[1]:
                label3[1] += 1
            else:
                label3[2] += 1

    width = 0.5
    fig, ax = plt.subplots()
    ax.bar(labels, label1, width, label=labels[0], color="#ABDEB6")
    ax.bar(labels, label2, width, label=labels[1], color="#5BBACF")
    ax.bar(labels, label3, width, label=labels[2], color="#298ABD")
    ax.set_ylabel('Predict Labels')
    ax.set_xlabel('Observed Labels')
    ax.set_title('Stacked Bar - ' + model)
    ax.legend()
    plt.savefig("img/%s-StackedBar.png" % model)
    plt.show()
