import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd

# Load dataset
data = pd.read_csv('diabetes.csv')
X = data.iloc[:, :-1].values  # Features
y = data.iloc[:, -1].values   # Labels

# Split into train (70%) and test (30%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Normalize data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

class NeuralNetwork:
    def __init__(self):
        # Updated: Hidden layer has 3 neurons now
        self.W1 = np.random.randn(8, 3) * 0.01  # Input layer (8 features) → Hidden layer (3 neurons)
        self.b1 = np.zeros((1, 3))
        self.W2 = np.random.randn(3, 1) * 0.01  # Hidden layer (3 neurons) → Output layer (1 neuron)
        self.b2 = np.zeros((1, 1))
    
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    def relu(self, x):
        return np.maximum(0, x)
    
    def relu_derivative(self, x):
        return np.where(x > 0, 1, 0)
    
    def forward(self, X):
        self.z1 = np.dot(X, self.W1) + self.b1  # Input to hidden layer
        self.a1 = self.relu(self.z1)  # ReLU activation
        self.z2 = np.dot(self.a1, self.W2) + self.b2  # Input to output layer
        self.a2 = self.sigmoid(self.z2)  # Sigmoid activation (output)
        return self.a2
    
    def backward(self, X, y, learning_rate):
        m = X.shape[0]
        
        #output layer error
        dZ2 = self.a2 - y.reshape(-1, 1)

        #gradients of loss
        dW2 = np.dot(self.a1.T, dZ2) / m
        db2 = np.sum(dZ2, axis=0, keepdims=True) / m
        
        #hidden layer error 
        dZ1 = np.dot(dZ2, self.W2.T) * self.relu_derivative(self.z1)

        #gradients of hidden layer
        dW1 = np.dot(X.T, dZ1) / m
        db1 = np.sum(dZ1, axis=0, keepdims=True) / m
        
        self.W1 -= learning_rate * dW1
        self.b1 -= learning_rate * db1
        self.W2 -= learning_rate * dW2
        self.b2 -= learning_rate * db2
    
    def train(self, X, y, epochs=500, learning_rate=0.3):
        for epoch in range(epochs):
            y_pred = self.forward(X)
            loss = -np.mean(y * np.log(y_pred + 1e-8) + (1 - y) * np.log(1 - y_pred + 1e-8))  # Numerical stability
            self.backward(X, y, learning_rate)
            if epoch % 100 == 0:
                print(f"Epoch {epoch}, Loss: {loss:.4f}")
    
    def predict(self, X, threshold=0.5):
        prob = self.forward(X)
        return (prob >= threshold).astype(int)

# Initialize and train network
nn = NeuralNetwork()
nn.train(X_train, y_train, epochs=1000, learning_rate=0.01)

# Predict and calculate accuracy
y_pred = nn.predict(X_test)
accuracy = np.mean(y_pred.flatten() == y_test)
print(f"Test Accuracy: {accuracy * 100:.2f}%")
