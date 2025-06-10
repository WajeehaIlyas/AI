import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load Data
def load_data():
    red_wine = pd.read_csv(
        'https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv',
        delimiter=';'
    )
    red_wine['type'] = 0  # 0 for red

    white_wine = pd.read_csv(
        'https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv',
        delimiter=';'
    )
    white_wine['type'] = 1  # 1 for white

    wine_data = pd.concat([red_wine, white_wine], axis=0)
    wine_data = wine_data.sample(frac=1, random_state=42).reset_index(drop=True)
    return wine_data

data = load_data()

# 1. Data Analysis and Visualization
print("\n--- Dataset Overview ---")
print(data.info())
print(data.describe())

# Class Balance
sns.countplot(x='type', data=data)
plt.title('Class Balance: Red vs White Wine')
plt.xticks([0, 1], ['Red', 'White'])
plt.show()

# Feature Distributions
data.hist(bins=30, figsize=(15, 10))
plt.tight_layout()
plt.show()

# Correlation Heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(data.corr(), annot=False, cmap='coolwarm')
plt.title("Feature Correlation Heatmap")
plt.show()

# 2. Preprocessing
# Normalize features (excluding 'type' which is the label)
X = data.drop(columns=['type']).values
y = data['type'].values

# Min-max normalization
X_min = X.min(axis=0)
X_max = X.max(axis=0)
X = (X - X_min) / (X_max - X_min)

# Train-test split
def train_test_split(X, y, test_size=0.2):
    split_idx = int(len(X) * (1 - test_size))
    return X[:split_idx], X[split_idx:], y[:split_idx], y[split_idx:]

X_train, X_test, y_train, y_test = train_test_split(X, y)

# 3. Logistic Regression Implementation
class LogisticRegressionScratch:
    def __init__(self, lr=0.1, epochs=1000):
        self.lr = lr
        self.epochs = epochs

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def fit(self, X, y):
        self.weights = np.zeros(X.shape[1])
        self.bias = 0

        for _ in range(self.epochs):
            linear_model = np.dot(X, self.weights) + self.bias
            y_predicted = self.sigmoid(linear_model)

            dw = np.dot(X.T, (y_predicted - y)) / y.size
            db = np.sum(y_predicted - y) / y.size

            self.weights -= self.lr * dw
            self.bias -= self.lr * db

    def predict(self, X):
        linear_model = np.dot(X, self.weights) + self.bias
        y_pred = self.sigmoid(linear_model)
        return [1 if i >= 0.5 else 0 for i in y_pred]

# Train the model
model = LogisticRegressionScratch(lr=0.5, epochs=3000)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Accuracy
def accuracy(y_true, y_pred):
    return np.mean(y_true == y_pred)

acc = accuracy(y_test, y_pred)
print(f"\n✅ Final Accuracy: {acc * 100:.2f}%")
