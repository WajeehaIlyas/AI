import numpy as np
from collections import Counter
import networkx as nx
import matplotlib.pyplot as plt

data = np.array([
    [2, 5, 0], [3, 6, 0], [5, 6, 1], [6, 5, 1], [8, 7, 1],
    [1, 4, 0], [4, 6, 1], [7, 8, 1], [2, 6, 0], [5, 5, 1]
])
X, y = data[:, :-1], data[:, -1]

def calculate_entropy(labels):
    counts = Counter(labels)
    total = len(labels)
    return -sum((count / total) * np.log2(count / total) for count in counts.values())

def find_optimal_split(X, y):
    optimal_feature, optimal_value, lowest_entropy = None, None, float('inf')
    n_samples, n_features = X.shape
    
    for feature in range(n_features):
        unique_values = np.unique(X[:, feature])
        for value in unique_values:
            left_mask = X[:, feature] <= value
            right_mask = ~left_mask
            
            left_labels, right_labels = y[left_mask], y[right_mask]
            left_entropy = calculate_entropy(left_labels)
            right_entropy = calculate_entropy(right_labels)
            
            weighted_entropy = (len(left_labels) * left_entropy + len(right_labels) * right_entropy) / n_samples
            
            if weighted_entropy < lowest_entropy:
                optimal_feature, optimal_value, lowest_entropy = feature, value, weighted_entropy
    
    return optimal_feature, optimal_value

class DecisionTreeClassifier:
    def __init__(self, max_depth=None):
        self.max_depth = max_depth
        self.tree = None
    
    def build_tree(self, X, y, depth=0):
        if len(set(y)) == 1:
            return y[0]
        if self.max_depth is not None and depth >= self.max_depth:
            return Counter(y).most_common(1)[0][0]
        
        feature, threshold = find_optimal_split(X, y)
        if feature is None:
            return Counter(y).most_common(1)[0][0]
        
        left_mask = X[:, feature] <= threshold
        right_mask = ~left_mask
        
        left_subtree = self.build_tree(X[left_mask], y[left_mask], depth + 1)
        right_subtree = self.build_tree(X[right_mask], y[right_mask], depth + 1)
        
        return {'feature': feature, 'threshold': threshold, 'left': left_subtree, 'right': right_subtree}
    
    def fit(self, X, y):
        self.tree = self.build_tree(X, y)
    
    def classify(self, node, x):
        if not isinstance(node, dict):
            return node
        
        if x[node['feature']] <= node['threshold']:
            return self.classify(node['left'], x)
        else:
            return self.classify(node['right'], x)
    
    def predict(self, X):
        return np.array([self.classify(self.tree, x) for x in X])

def construct_graph(node, graph=None, parent=None, edge_label=''):
    if graph is None:
        graph = nx.DiGraph()
    
    node_label = f"Feature {node['feature']} <= {node['threshold']}" if isinstance(node, dict) else f"Class: {node}"
    node_id = len(graph.nodes)
    graph.add_node(node_id, label=node_label)
    
    if parent is not None:
        graph.add_edge(parent, node_id, label=edge_label)
    
    if isinstance(node, dict):
        construct_graph(node['left'], graph, node_id, 'Yes')
        construct_graph(node['right'], graph, node_id, 'No')
    
    return graph

def visualize_tree(tree):
    graph = construct_graph(tree)
    pos = nx.spring_layout(graph)
    labels = nx.get_node_attributes(graph, 'label')
    edge_labels = nx.get_edge_attributes(graph, 'label')
    
    plt.figure(figsize=(10, 6))
    nx.draw(graph, pos, with_labels=True, labels=labels, node_size=3000, node_color='lightblue')
    nx.draw_networkx_edge_labels(graph, pos, edge_labels=edge_labels)
    plt.show()

dt = DecisionTreeClassifier(max_depth=3)
dt.fit(X, y)
predictions = dt.predict(X)
print("Predictions:", predictions)
visualize_tree(dt.tree)
