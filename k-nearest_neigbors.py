import numpy as np
from math import sqrt
from collections import Counter
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

def euclidean_distance(point1, point2):
    return sqrt(sum((p1 - p2) ** 2 for p1, p2 in zip(point1, point2)))

def get_k_neighbors(X_train, y_train, test_sample, k):
    distances = [(euclidean_distance(test_sample, X_train[i]), y_train[i]) for i in range(len(X_train))]
    distances.sort(key=lambda x: x[0])
    return [label for _, label in distances[:k]]

def predict_classification(X_train, y_train, test_sample, k):
    neighbors = get_k_neighbors(X_train, y_train, test_sample, k)
    return Counter(neighbors).most_common(1)[0][0] 

def evaluate_knn(X_train, X_test, y_train, y_test, k):
    predictions = [predict_classification(X_train, y_train, sample, k) for sample in X_test]
    accuracy = sum(p == y for p, y in zip(predictions, y_test)) / len(y_test)
    return accuracy

iris = load_iris()
X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

for k in [1, 3, 5, 7]:
    accuracy = evaluate_knn(X_train, X_test, y_train, y_test, k)
    print(f'Accuracy for k={k}: {accuracy:.2f}')
