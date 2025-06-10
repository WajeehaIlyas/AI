import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

# Load dataset
iris = load_iris()
X = iris.data
y = iris.target

# Standardize data
means = np.mean(X, axis=0)
stds = np.std(X, axis=0)
X_std = (X - means) / stds

# Correlation matrix
n = X_std.shape[0]
correlation_matrix = (1 / (n - 1)) * np.dot(X_std.T, X_std)

# Covariance matrix (same as correlation matrix here)
cov_matrix = correlation_matrix

# Eigenvalues and eigenvectors
eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
sorted_indices = np.argsort(eigenvalues)[::-1]
eigenvalues_sorted = eigenvalues[sorted_indices]
eigenvectors_sorted = eigenvectors[:, sorted_indices]

# PCA projection
X_pca = np.dot(X_std, eigenvectors_sorted[:, :2])

# Explained variance ratio
explained_variance_ratio = eigenvalues_sorted / np.sum(eigenvalues_sorted)

# Plot
colors = ['r', 'g', 'b']
species = iris.target_names
plt.figure(figsize=(8, 6))
for i, color in enumerate(colors):
    plt.scatter(X_pca[y == i, 0], X_pca[y == i, 1], label=species[i], c=color)
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('PCA of IRIS Dataset (Manual Implementation)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Output results
print("Correlation Matrix (4x4):\n", correlation_matrix.round(4))
print("\nEigenvalues:\n", eigenvalues_sorted.round(4))
print("\nExplained Variance Ratio:\n", explained_variance_ratio.round(4))
print("\nTop 2 Principal Components (first 5 rows):\n", X_pca[:5].round(4))
