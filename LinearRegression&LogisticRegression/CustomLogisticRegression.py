import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class BinomialLogisticRegression:

    def __init__(self, learning_rate = 0.01, epochs = 1000, critical_value = 0.5):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.critical_value = critical_value
        self.col = 0
        self.weight = None
        self.bias = 0
        self.cost = []
        self.score = {"Accuracy": [], "Recall": [], "Precision": [], "F1": []}

    def get_sigmoid(self, z):
        sigmoid = 1 / (1 + np.exp(-z))
        return sigmoid
    
    def compute_cost(self, y_sig, y_train):
        cost = -(np.dot(y_train, np.log(y_sig)) + np.dot((1 - y_train), np.log(y_sig))) / len(y_train)
        return cost

    def fit(self, X_train, y_train, log = False):
        X_train = np.array(X_train)
        y_train = np.array(y_train)
        self.col = X_train.shape[1]
        self.weight = [0] * self.col
        for i in range(self.epochs):
            y_linear = np.dot(self.weight, X_train.T) + self.bias
            y_sig = self.get_sigmoid(y_linear)
            y_curr = np.where(y_sig >= 0.5, 1, 0)
            self.cost.append(self.compute_cost(y_sig, y_train))
            d_weight = np.dot((y_curr - y_train).T, X_train) / len(y_train)
            d_bias = (y_curr - y_train).sum() / len(y_train)
            self.weight -= np.dot(d_weight, self.learning_rate) * self.cost[i]
            self.bias -= d_bias * self.learning_rate * self.cost[i]
            
            true_positive = np.count_nonzero((y_train == y_curr) & y_train == 1)
            true_negative = np.count_nonzero((y_train == y_curr) & y_train == 0)
            false_positive = np.count_nonzero((y_train != y_curr) & y_train == 1)
            false_negative = np.count_nonzero((y_train != y_curr) & y_train == 0)
            
            self.score["Accuracy"].append((true_positive + true_negative) / (true_positive + true_negative + false_positive + false_negative))
            recall = (true_positive) / (true_positive + false_negative)
            self.score["Recall"].append(recall)
            precision = (true_positive) / (true_positive + false_positive)
            self.score["Precision"].append(precision)
            try:
                self.score["F1"].append(2 * (precision * recall) / (precision + recall))
            except:
                self.score["F1"].append(0)
            if log == True:
                print("[Epoch %d] Accuracy Score : %f" %(i, self.score["Accuracy"][i]))
    
    def get_score(self, method = 'Accuracy'):
        if method == "Accuracy":
            print("Accuracy Score :", self.score["Accuracy"][-1])
        elif method == "Recall":
            print("Recall Score :", self.score["Recall"][-1])
        elif method == "Precision":
            print("Precision Score :", self.score["Precision"][-1])
        elif method == "F1":
            print("F1 Score :", self.score["F1"][-1])
        else:
            print("Not Supported")
            print("Support only MSE, MAE, MAPE, RMSE, R2")

    def predict(self, X_test):
        y_linear = np.dot(self.weight, X_test.T) + self.bias
        y_sig = self.get_sigmoid(y_linear)
        y_pred = np.where(y_sig >= 0.5, 1, 0)
        return y_pred