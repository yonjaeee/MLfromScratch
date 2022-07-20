import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class OLSLSingleinearRegression:

    def __init__(self):
        self.weight = 0
        self.bias = 0
        self.score = dict()
        
    def fit(self, X_train, y_train, graph = False):
        X_train = np.array(X_train)
        X_mean = np.mean(X_train)
        y_mean = np.mean(y_train)
        self.weight = ((X_train - X_mean) * (y_train - y_mean)).sum() / ((X_train - X_mean) ** 2).sum()
        self.bias = y_mean - self.weight * X_mean
        self.score["MAE"] = abs(y_train - y_mean).sum() / len(y_train)
        self.score["MSE"] = ((y_train - y_mean)**2).sum() / len(y_train)
        self.score["RMSE"] = (((y_train - y_mean)**2).sum() / len(y_train)) ** (1/2) 
        if graph == True:
            plt.scatter(X_train, y_train)
            plt.plot(X_train, self.weight*X_train + self.bias, color = 'red')
            plt.title('OLS Single Linear Regression')
            plt.show()
    
    def score(self, method = "MSE"):
        if method == "MSE":
            print("MSE Score :", self.score["MSE"])
        elif method == "MAE":
            print("MAE Score :", self.score["MAE"])
        elif method == "RMSE":
            print("RMSE Score :", self.score["RMSE"])
        else:
            print("Not Supported")
            print("Support only MSE, MAE, RMSE")

    def predict(self, X_test):
        y_pred = self.weight * X_test + self.bias
        return y_pred
class OLSLMultiLinearRegression:

    def __init__(self):
        self.col = 0
        self.weight = None
        self.bias = 0
        self.score = dict()
        
    def fit(self, X_train, y_train):
        X_train = np.array(X_train)
        X_mean = np.mean(X_train, axis = 0)
        y_mean = np.mean(y_train)
        self.col = X_train.shape[1]
        self.weight = np.dot(np.dot(np.linalg.inv(np.dot(X_train.T, X_train)), X_train.T), y_train)
        self.bias = y_mean - np.dot(self.weight, X_mean)
        self.score["MAE"] = abs(y_train - y_mean).sum() / len(y_train)
        self.score["MSE"] = ((y_train - y_mean)**2).sum() / len(y_train)
        self.score["RMSE"] = (((y_train - y_mean)**2).sum() / len(y_train)) ** (1/2) 
    
    def get_score(self, method = "MSE"):
        if method == "MSE":
            print("MSE Score :", self.score["MSE"])
        elif method == "MAE":
            print("MAE Score :", self.score["MAE"])
        elif method == "RMSE":
            print("RMSE Score :", self.score["RMSE"])
        else:
            print("Not Supported")
            print("Support only MSE, MAE, RMSE")

    def predict(self, X_test):
        y_pred = np.dot(self.weight, X_test.T) + self.bias
        return y_pred