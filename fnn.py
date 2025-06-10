import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_heart_data():
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
    column_names = [
        "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", 
        "thalach", "exang", "oldpeak", "slope", "ca", "thal", "target"
    ]
    data = pd.read_csv(url, names=column_names, na_values="?")
    data["target"] = data["target"].apply(lambda x: 1 if x > 0 else 0)
    data.dropna(inplace=True)
    X = data.drop("target", axis=1).values
    y = data["target"].values.reshape(-1, 1)
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    return X, y

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def sigmoid_derivative(a):
    return a * (1 - a)

def initialize_parameters(input_size, hidden_size, output_size):
    np.random.seed(1)
    return {
        'W1': np.random.randn(input_size, hidden_size) * 0.01,
        'b1': np.zeros((1, hidden_size)),
        'W2': np.random.randn(hidden_size, output_size) * 0.01,
        'b2': np.zeros((1, output_size))
    }

def forward_pass(X, params, apply_dropout=False, keep_prob=0.9):
    Z1 = np.dot(X, params['W1']) + params['b1']
    A1 = sigmoid(Z1)
    
    dropout_mask = None
    if apply_dropout:
        dropout_mask = (np.random.rand(*A1.shape) < keep_prob)
        A1 *= dropout_mask
        A1 /= keep_prob
    
    Z2 = np.dot(A1, params['W2']) + params['b2']
    A2 = sigmoid(Z2)
    
    return A2, {'A1': A1, 'A2': A2, 'Z1': Z1, 'Z2': Z2, 'dropout_mask': dropout_mask}

def compute_loss(y_true, y_pred):
    eps = 1e-8
    return -np.mean(y_true * np.log(y_pred + eps) + (1 - y_true) * np.log(1 - y_pred + eps))

def compute_accuracy(y_true, y_pred):
    return np.mean((y_pred > 0.5) == y_true)

def backward_pass(X, y, cache, params, learning_rate, keep_prob=1.0):
    A1, A2 = cache['A1'], cache['A2']
    m = X.shape[0]
    
    dZ2 = A2 - y
    dW2 = np.dot(A1.T, dZ2) / m
    db2 = np.sum(dZ2, axis=0, keepdims=True) / m
    
    dA1 = np.dot(dZ2, params['W2'].T)
    dA1 *= sigmoid_derivative(A1)
    
    if cache['dropout_mask'] is not None:
        dA1 *= cache['dropout_mask']
        dA1 /= keep_prob
    
    dW1 = np.dot(X.T, dA1) / m
    db1 = np.sum(dA1, axis=0, keepdims=True) / m
    
    params['W2'] -= learning_rate * dW2
    params['b2'] -= learning_rate * db2
    params['W1'] -= learning_rate * dW1
    params['b1'] -= learning_rate * db1

def train(X, y, hidden_size=10, epochs=100, learning_rate=0.1, keep_prob=0.8, patience=5):
     input_size = X.shape[1]
     output_size = 1
    
     X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
     params = initialize_parameters(input_size, hidden_size, output_size)
     #best validation loss for early stopping
     best_loss = float('inf')
     #count number of epochs without improvement
     wait = 0
     history = {'loss': [], 'acc': [], 'val_loss': [], 'val_acc': []}
    
     for epoch in range(epochs):
         y_pred, cache = forward_pass(X_train, params, apply_dropout=False, keep_prob=keep_prob)
         loss = compute_loss(y_train, y_pred)
         acc = compute_accuracy(y_train, y_pred)
        
         backward_pass(X_train, y_train, cache, params, learning_rate, keep_prob)
        
         val_pred, _ = forward_pass(X_val, params, apply_dropout=False)
         val_loss = compute_loss(y_val, val_pred)
         val_acc = compute_accuracy(y_val, val_pred)
        
         history['loss'].append(loss)
         history['acc'].append(acc)
         history['val_loss'].append(val_loss)
         history['val_acc'].append(val_acc)
        
         print(f"Epoch {epoch + 1}/{epochs} - Loss: {loss:.4f} - Acc: {acc:.4f} - Val Loss: {val_loss:.4f} - Val Acc: {val_acc:.4f}")
        
         if val_loss < best_loss:
             best_loss = val_loss
             wait = 0
         else:
             wait += 1
             if wait >= patience:
                 print("⏹ Early stopping triggered.")
                 break
    
     return params, history

def plot_metrics(history):
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history['loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Val Loss')
    plt.legend()
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Loss Curve')
    
    plt.subplot(1, 2, 2)
    plt.plot(history['acc'], label='Train Acc')
    plt.plot(history['val_acc'], label='Val Acc')
    plt.legend()
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Accuracy Curve')
    
    plt.show()

if __name__ == "__main__":
    X, y = load_heart_data()
    params, history = train(X, y, hidden_size=10, epochs=200, learning_rate=0.1, keep_prob=0.8, patience=5)
    plot_metrics(history)