#!/usr/bin/env python3
# encoding: utf-8
"""
@author: Daniel Fuchs

CS3001: Data Science - Homework #5: K-Nearest-Neighbors
"""
import pickle
from sklearn import metrics
from sklearn.datasets import load_iris


def main():
    iris = load_iris()
    test(iris.data, iris.target)


def test(x, y, model_file='model.txt', output_file='results.txt'):
    model = pickle.load(open(model_file, 'rb'))
    predictions = model.predict(x)
    accuracy = metrics.accuracy_score(y, predictions)
    with open(output_file, 'w+') as f:
        f.write('Accuracy: %f' % accuracy)


def get_accuracy(x, y, model_file='model.txt'):
    model = pickle.load(open(model_file, 'rb'))
    predictions = model.predict(x)
    return metrics.accuracy_score(y, predictions)


if __name__ == '__main__':
    main()
