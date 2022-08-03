import math
import numpy as np
import pandas as pd

class DecisionTreeNode:

    def __init__(self, impurity, predicted_class):
        self.impurity = impurity
        self.predicted_class = predicted_class
        self.feature_index = None
        self.threshold = None
        self.left = None
        self.right = None

class DecisionTreeClassifier:
    
    def __init__(self, min_samples_split = 20, max_depth = 5):
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        
    def fit(self, X, y):
        self.X = X
        self.y = y
        self.n_classes = len(set(y))
        self.classes = set(y)
        self.n_features = X.shape[1]
        self.tree = self.build_tree(X, y)

    def best_split(self, X, y):
        if len(y) <= 1:
            return None, None

        num_parent = [np.sum(y == c) for c in self.classes]
        best_gini = self.gini_impurity(num_parent)
        best_idx, best_thr = None, None

        for idx in range(self.n_features):
            thresholds, classes = zip(*sorted(zip(X[:, idx], y)))
            num_left = [0] * self.n_classes
            num_right = num_parent.copy()
            for i in range(1, len(y)):
                c = classes[i - 1]
                num_left[c] += 1
                num_right[c] -= 1
                gini_left = self.gini_impurity(num_left)
                gini_right = self.gini_impurity(num_left)
                weighted_gini = (i * gini_left + (len(y) - i) * gini_right) / len(y)
                if thresholds[i] == thresholds[i - 1]:
                    continue
                if weighted_gini < best_gini:
                    best_gini = weighted_gini
                    best_idx = idx
                    best_thr = thresholds[i]

        return best_idx, best_thr

    def gini_impurity(self, nums):
        total_count = sum(nums)
        if total_count == 0:
            return 0
        else:
            gini =  1.0 - sum((x / total_count) ** 2 for x in nums)
            return gini
    
    def entropy(self, nums):
        total_count = sum(nums)
        if total_count == 0:
            return 0
        else:
            entropy =  1.0 - sum((x/total_count) * math.log2(x/total_count) for x in nums)
            return entropy

    def build_tree(self, X, y, depth = 0):
        gini = self.gini_impurity([np.sum(y == c) for c in self.classes])
        num_samples_per_class = [np.sum(y == i) for i in range(self.n_classes)]
        predicted_class = np.argmax(num_samples_per_class)
        node = DecisionTreeNode(impurity=gini, predicted_class=predicted_class)
        if depth < self.max_depth:
            idx, thr = self.best_split(X, y)
            if idx != None:
                idx_left = X[:, idx] < thr
                X_left, y_left = X[idx_left], y[idx_left]
                X_right, y_right = X[~idx_left], y[~idx_left]
                node.feature_index = idx
                node.threshold = thr
                node.left = self.build_tree(X_left, y_left, depth + 1)
                node.right = self.build_tree(X_right, y_right, depth + 1)
        return node
    
    def predict(self, X):
        return [self.predict_each_row(inputs) for inputs in X]

    def predict_each_row(self, inputs):
        node = self.tree
        while node.left:
            if inputs[node.feature_index] < node.threshold:
                node = node.left
            else:
                node = node.right
        return node.predicted_class