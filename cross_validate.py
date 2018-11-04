#!/usr/bin/env python3
# encoding: utf-8
"""
@author: Daniel Fuchs

CS3001: Data Science - Homework #5: K-Nearest-Neighbors
"""
import random
import numpy as np
from training import train
from testing import get_accuracy
from sklearn.datasets import load_iris


def main():
    iris = load_iris()
    avg_accuracy = cross_validate(iris.data, iris.target, 'DECISION_TREE')
    print("Average Accuracy for Decision Tree:", avg_accuracy)


def evaluate_avg_cv_accuracy(model_name, neighbors=10, num_partitions=5, iterations=100):
    avg_acc = 0
    iris = load_iris()
    for j in range(iterations):
        avg_acc += cross_validate(iris.data, iris.target, model_name=model_name, neighbors=neighbors)
    return avg_acc / iterations


def cross_validate(x, y, model_name, neighbors=10, num_partitions=5):
    data_set = np.append(x, np.split(y, len(x)), axis=1)

    partition = []
    for i in range(num_partitions):
        partition.append([])

    for i in range(min(len(x), len(y))):
        random_part = random.randrange(0, num_partitions, 1)
        partition[random_part].append(data_set[i])

    for i in range(num_partitions):
        partition[i] = np.array(partition[i])
    partition = np.array(partition)

    accuracy = 0
    for i in range(num_partitions):
        testing_set = partition[i]
        t_sets = []
        for j in range(num_partitions):
            if j != i:
                t_sets.append(partition[j])
        training_set = np.array(t_sets.pop())
        for t_set in t_sets:
            training_set = np.append(training_set, t_set, axis=0)

        training_data = training_set[:, :-1]
        training_target = training_set[:, -1]
        testing_data = testing_set[:, :-1]
        testing_target = testing_set[:, -1]

        train(training_data, training_target, model_name=model_name, neighbors=neighbors)
        accuracy += get_accuracy(testing_data, testing_target)
    return accuracy / num_partitions


if __name__ == '__main__':
    main()
