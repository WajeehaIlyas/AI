import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

def min_max_scaling(X):
    return (X - X.min()) / (X.max() - X.min())

def linear_regression(X, y):
    n = len(X)
    X_mean = np.mean(X)
    y_mean = np.mean(y)
    
    numerator = np.sum((X - X_mean) * (y - y_mean))
    denominator = np.sum((X - X_mean) ** 2)
    m = numerator / denominator
    c = y_mean - m * X_mean
    
    return m, c

def mean_squared_error(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

df = pd.read_csv('dataset.csv')
X = df['X1'].values
y = df['y'].values

X_scaled = min_max_scaling(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

m, c = linear_regression(X_train, y_train)

y_pred_train = m * X_train + c
y_pred_test = m * X_test + c

plt.figure(figsize=(10, 5))

plt.scatter(X_train, y_train, color='blue', label='Training Data')
plt.plot(X_train, y_pred_train, color='red', label='Regression Line')
plt.xlabel('Feature X1')
plt.ylabel('Target y')
plt.title('Linear Regression - Training Data')
plt.legend()
plt.show()

plt.figure(figsize=(10, 5))
plt.scatter(X_test, y_test, color='green', label='Testing Data')
plt.plot(X_test, y_pred_test, color='red', label='Regression Line')
plt.xlabel('Feature X1')
plt.ylabel('Target y')
plt.title('Linear Regression - Testing Data')
plt.legend()
plt.show()

mse = mean_squared_error(y_test, y_pred_test)
print(f'Mean Squared Error (MSE): {mse:.4f}')
