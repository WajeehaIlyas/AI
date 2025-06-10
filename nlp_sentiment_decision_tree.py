import string
import random
from collections import Counter, defaultdict

# Sample data
sample_positive = [
    "i love this product it works great",
    "excellent service and fast delivery",
    "best purchase i have made this year",
    "very happy with the quality of this item",
    "the customer service was amazing and helpful",
    "works perfectly and arrived on time",
    "exceeded my expectations highly recommend",
    "easy to use and does exactly what it says",
    "great value for money will buy again",
    "fantastic product and excellent experience"
]

sample_negative = [
    "terrible experience would not recommend",
    "product broke after two days",
    "customer service was unhelpful and rude",
    "waste of money does not work properly", 
    "extremely disappointed with this purchase",
    "arrived late and damaged",
    "did not match the description at all",
    "poor quality and overpriced",
    "worst product i have ever bought",
    "completely useless and frustrating"
]

# Stop words
stop_words = ["the", "a", "an", "and", "or", "but", "is", "are", "was", "were", 
              "i", "you", "he", "she", "it", "we", "they", "this", "that", "have", "has"]

# Preprocessing functions
def lowercase(text):
    return text.lower()

def remove_punctuation(text):
    return ''.join(c for c in text if c.isalnum() or c.isspace())

def tokenize(text):
    return text.split()

def remove_stop_words(tokens, stop_words_list):
    return [token for token in tokens if token not in stop_words_list]

def preprocess_text(text):
    text = lowercase(text)
    text = remove_punctuation(text)
    tokens = tokenize(text)
    tokens = remove_stop_words(tokens, stop_words)
    return tokens

# Frequency counter
def count_frequencies(docs):
    pos_tokens = [token for doc in docs[0] for token in preprocess_text(doc)]
    neg_tokens = [token for doc in docs[1] for token in preprocess_text(doc)]
    print("Top Positive Words:", Counter(pos_tokens).most_common(10))
    print("Top Negative Words:", Counter(neg_tokens).most_common(10))

# Vocabulary
def build_vocabulary(documents):
    vocab = set()
    for doc in documents:
        vocab.update(doc)
    return sorted(list(vocab))

# Bag-of-Words
def create_bow_vector(document, vocabulary):
    vec = [0] * len(vocabulary)
    word_counts = Counter(document)
    for i, word in enumerate(vocabulary):
        if word in word_counts:
            vec[i] = word_counts[word]
    return vec

# Binary Feature Encoding
def convert_to_binary_features(bow_vectors):
    return [[1 if count > 0 else 0 for count in vec] for vec in bow_vectors]

# Decision Tree Node
class Node:
    def __init__(self, feature_idx=None, left=None, right=None, value=None):
        self.feature_idx = feature_idx
        self.left = left
        self.right = right
        self.value = value

# Gini and Info Gain
def calculate_gini(y):
    total = len(y)
    if total == 0:
        return 0
    counts = Counter(y)
    return 1 - sum((c / total) ** 2 for c in counts.values())

def calculate_information_gain(parent, left, right):
    p_len, l_len, r_len = len(parent), len(left), len(right)
    return calculate_gini(parent) - (l_len/p_len)*calculate_gini(left) - (r_len/p_len)*calculate_gini(right)

# Best Split Finder
def find_best_split(X, y):
    best_gain = 0
    best_idx = None
    for idx in range(len(X[0])):
        left_y, right_y = [], []
        for xi, yi in zip(X, y):
            if xi[idx] == 0:
                left_y.append(yi)
            else:
                right_y.append(yi)
        gain = calculate_information_gain(y, left_y, right_y)
        if gain > best_gain:
            best_gain = gain
            best_idx = idx
    return best_idx

# Tree Builder
def build_tree(X, y, depth=0, max_depth=5, min_samples_split=2):
    if len(set(y)) == 1:
        return Node(value=y[0])
    if len(y) < min_samples_split or depth == max_depth:
        return Node(value=Counter(y).most_common(1)[0][0])
    best_idx = find_best_split(X, y)
    if best_idx is None:
        return Node(value=Counter(y).most_common(1)[0][0])
    left_X, left_y, right_X, right_y = [], [], [], []
    for xi, yi in zip(X, y):
        if xi[best_idx] == 0:
            left_X.append(xi)
            left_y.append(yi)
        else:
            right_X.append(xi)
            right_y.append(yi)
    left_node = build_tree(left_X, left_y, depth+1, max_depth, min_samples_split)
    right_node = build_tree(right_X, right_y, depth+1, max_depth, min_samples_split)
    return Node(feature_idx=best_idx, left=left_node, right=right_node)

# Predict Functions
def predict_sample(x, tree):
    if tree.value is not None:
        return tree.value
    if x[tree.feature_idx] == 0:
        return predict_sample(x, tree.left)
    else:
        return predict_sample(x, tree.right)

def decision_tree_predict(X, tree):
    return [predict_sample(x, tree) for x in X]

# Train-Test Split
def split_data(data, labels, train_ratio=0.8):
    #combine data and labels
    combined = list(zip(data, labels))
    #shuffle the combined data
    random.shuffle(combined)
    split = int(train_ratio * len(data))
    train = combined[:split]
    test = combined[split:]
    return ([x[0] for x in train], [x[1] for x in train],
            [x[0] for x in test], [x[1] for x in test])

# Evaluation
def evaluate(predictions, labels):
    correct = sum(1 for p, l in zip(predictions, labels) if p == l)
    return correct / len(labels) if labels else 0

# Tree Visualization
def visualize_tree(node, feature_names, depth=0):
    indent = "  " * depth
    if node.value is not None:
        print(f"{indent}Leaf: {node.value}")
    else:
        feature = feature_names[node.feature_idx]
        print(f"{indent}Feature: {feature}")
        visualize_tree(node.left, feature_names, depth+1)
        visualize_tree(node.right, feature_names, depth+1)

# Feature Importance
def get_feature_importance(tree, feature_names, importance=None):
    if importance is None:
        importance = defaultdict(int)
    if tree is None or tree.value is not None:
        return importance
    importance[feature_names[tree.feature_idx]] += 1
    get_feature_importance(tree.left, feature_names, importance)
    get_feature_importance(tree.right, feature_names, importance)
    return sorted(importance.items(), key=lambda x: -x[1])

# Main Program
def main():
    all_docs = sample_positive + sample_negative
    labels = ["positive"] * len(sample_positive) + ["negative"] * len(sample_negative)
    
    # Preprocessing
    preprocessed = [preprocess_text(doc) for doc in all_docs]
    
    # Frequency Analysis
    count_frequencies((sample_positive, sample_negative))
    
    # Vocabulary and BOW + Binary
    vocab = build_vocabulary(preprocessed)
    vectors = [create_bow_vector(doc, vocab) for doc in preprocessed]
    binary_vectors = convert_to_binary_features(vectors)
    
    # Split
    X_train, y_train, X_test, y_test = split_data(binary_vectors, labels)
    
    # Train
    tree = build_tree(X_train, y_train, max_depth=4)
    
    # Predict
    preds = decision_tree_predict(X_test, tree)
    print("\nTest Predictions:", preds)
    
    # Evaluation
    accuracy = evaluate(preds, y_test)
    print("Accuracy:", accuracy)
    
    # Tree Structure
    print("\nDecision Tree Structure:")
    visualize_tree(tree, vocab)

    # Feature Importance
    print("\nImportant Features:")
    for word, score in get_feature_importance(tree, vocab):
        print(f"{word}: {score}")

if __name__ == "__main__":
    main()
