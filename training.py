#!/usr/bin/env python3
# encoding: utf-8
"""
@author: Daniel Fuchs

CS3001: Data Science - Homework #5: K-Nearest-Neighbors
"""
import sys
import pickle
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier


def main():
    iris = load_iris()
    train(iris.data, iris.target, 'DECISION_TREE')


def train(x, y, model_name, neighbors=10, output_file='model.txt'):
    if model_name.upper() == 'DECISION_TREE':
        classifier = DecisionTreeClassifier()
    elif model_name.upper() == 'KNN':
        classifier = KNeighborsClassifier(n_neighbors=neighbors)
    else:
        sys.exit("Invalid Classifier Type")

    model = classifier.fit(x, y)
    pickle.dump(model, open(output_file, 'wb'))


def get_model(x, y, model_name, neighbors=10):
    if model_name.upper() == 'DECISION_TREE':
        classifier = DecisionTreeClassifier()
    elif model_name.upper() == 'KNN':
        classifier = KNeighborsClassifier(n_neighbors=neighbors)
    else:
        sys.exit("Invalid Classifier Type")
    return classifier.fit(x, y)


if __name__ == '__main__':
    main()
