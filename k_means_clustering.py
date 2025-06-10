import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
data = pd.read_csv(url, sep=';')
X = data[['alcohol', 'volatile acidity']].values

def initialize_centroids(X, k):
    return X[random.sample(range(X.shape[0]), k)]

def assign_clusters(X, centroids):
    distances = np.linalg.norm(X[:, np.newaxis] - centroids, axis=2)
    return np.argmin(distances, axis=1)

def update_centroids(X, labels, k):
    return np.array([X[labels == i].mean(axis=0) for i in range(k)])

def k_means(X, k, max_iters=100, tol=1e-4):
    centroids = initialize_centroids(X, k)
    for _ in range(max_iters):
        labels = assign_clusters(X, centroids)
        new_centroids = update_centroids(X, labels, k)
        if np.linalg.norm(new_centroids - centroids) < tol:
            break
        centroids = new_centroids
    cluster_counts = np.bincount(labels, minlength=k)
    return centroids, labels, cluster_counts

k = 5  
final_centroids, cluster_labels, cluster_counts = k_means(X, k)

plt.figure(figsize=(8, 6))
for i in range(k):
    plt.scatter(X[cluster_labels == i, 0], X[cluster_labels == i, 1], label=f'Cluster {i}')
plt.scatter(final_centroids[:, 0], final_centroids[:, 1], color='red', marker='x', s=200, label='Centroids')
plt.xlabel('Alcohol')
plt.ylabel('Volatile Acidity')
plt.legend()
plt.title('K-Means Clustering')
plt.show()

print("Final Centroids:", final_centroids)
print("Cluster Assignments:", cluster_labels)
print("Number of Data Points in Each Cluster:", cluster_counts)