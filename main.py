#!/usr/bin/env python3
# encoding: utf-8
"""
@author: Daniel Fuchs

CS3001: Data Science - Homework #5: K-Nearest-Neighbors
"""
import matplotlib.pyplot as plt
from cross_validate import evaluate_avg_cv_accuracy
plt.rcParams['figure.figsize'] = (15, 5)


def main():
    # Show average accuracy of K-values
    scores = []
    for i in range(1, 101):
        scores.append(evaluate_avg_cv_accuracy('KNN', neighbors=i, iterations=100))
    plt.bar(range(1, 101), scores)
    plt.title("Cross-validation Accuracy of KNN for 'k'")
    plt.ylim(0.7)
    plt.xlabel("K-Value")
    plt.ylabel("Accuracy")
    plt.show()

    # Compare KNN and Decision Tree
    best_k_accuracy = evaluate_avg_cv_accuracy('KNN', neighbors=12, iterations=1000)
    dt_accuracy = evaluate_avg_cv_accuracy('DECISION_TREE', iterations=1000)
    print("Accuracy of KNN with K=12:", best_k_accuracy)
    print("Accuracy of Decision Tree:", dt_accuracy)


if __name__ == '__main__':
    main()
