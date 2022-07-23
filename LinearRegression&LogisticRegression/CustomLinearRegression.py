import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class OLSLSingleinearRegression:

    def __init__(self):
        self.weight = 0
        self.bias = 0
        self.score = {"MAE": [], "MAPE": [], "MSE": [], "RMSE": [], "R2": []}
        
    def fit(self, X_train, y_train, lr_graph = False):
        X_train = np.array(X_train)
        X_mean = np.mean(X_train)
        y_mean = np.mean(y_train)
        self.weight = ((X_train - X_mean) * (y_train - y_mean)).sum() / ((X_train - X_mean) ** 2).sum()
        self.bias = y_mean - self.weight * X_mean
        y_curr = self.weight*X_train + self.bias
        self.score["MAE"].append(abs(y_train - y_curr).sum() / len(y_train))
        self.score["MAPE"].append(abs((y_train - y_curr) / y_train).sum() / len(y_train) * 100)
        self.score["MSE"].append(((y_train - y_curr)**2).sum() / len(y_train))
        self.score["RMSE"].append((((y_train - y_curr)**2).sum() / len(y_train)) ** (1/2))
        self.score["R2"].append((np.corrcoef(y_train, y_curr)[0, 1]) ** 2)
        if lr_graph == True:
            plt.scatter(X_train, y_train)
            plt.plot(X_train, self.weight*X_train + self.bias, color = 'red')
            plt.title('OLS Single Linear Regression')
            plt.show()
    
    def get_score(self, method = "MSE"):
        if method == "MSE":
            print("MSE Score :", self.score["MSE"][-1])
        elif method == "MAE":
            print("MAE Score :", self.score["MAE"][-1])
        elif method == "MAPE":
            print("MAPE Score :", self.score["MAPE"][-1])
        elif method == "RMSE":
            print("RMSE Score :", self.score["RMSE"][-1])
        elif method == "R2":
            print("R2 Score :", self.score["R2"][-1])
        else:
            print("Not Supported")
            print("Support only MSE, MAE, MAPE, RMSE, R2")

    def predict(self, X_test):
        y_pred = self.weight * X_test + self.bias
        return y_pred

class OLSLMultiLinearRegression:

    def __init__(self):
        self.col = 0
        self.weight = None
        self.bias = 0
        self.score = {"MAE": [], "MAPE": [], "MSE": [], "RMSE": [], "R2": []}
        
    def fit(self, X_train, y_train):
        X_train = np.array(X_train)
        X_mean = np.mean(X_train, axis = 0)
        y_mean = np.mean(y_train)
        self.col = X_train.shape[1]
        self.weight = np.dot(np.dot(np.linalg.inv(np.dot(X_train.T, X_train)), X_train.T), y_train)
        self.bias = y_mean - np.dot(self.weight, X_mean)
        y_curr = np.dot(self.weight, X_train.T) + self.bias
        self.score["MAE"].append(abs(y_train - y_curr).sum() / len(y_train))
        self.score["MAPE"].append(abs((y_train - y_curr) / y_train).sum() / len(y_train) * 100)
        self.score["MSE"].append(((y_train - y_curr)**2).sum() / len(y_train))
        self.score["RMSE"].append((((y_train - y_curr)**2).sum() / len(y_train)) ** (1/2))
        self.score["R2"].append((np.corrcoef(y_train, y_curr)[0, 1]) ** 2)
    
    def get_score(self, method = "MSE"):
        if method == "MSE":
            print("MSE Score :", self.score["MSE"][-1])
        elif method == "MAE":
            print("MAE Score :", self.score["MAE"][-1])
        elif method == "MAPE":
            print("MAPE Score :", self.score["MAPE"][-1])
        elif method == "RMSE":
            print("RMSE Score :", self.score["RMSE"][-1])
        elif method == "R2":
            print("R2 Score :", self.score["R2"][-1])
        else:
            print("Not Supported")
            print("Support only MSE, MAE, RMSE R2")

    def predict(self, X_test):
        y_pred = np.dot(self.weight, X_test.T) + self.bias
        return y_pred

class GDSingleLinearRegression:

    def __init__(self, learning_rate = 0.01, epochs = 1000):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.weight = 0
        self.bias = 0
        self.score = {"MAE": [], "MAPE": [], "MSE": [], "RMSE": [], "R2": []}

    def fit(self, X_train, y_train, log = False, lr_graph = False):
        X_train = np.array(X_train)
        for i in range(self.epochs):
            y_curr = self.weight*X_train + self.bias
            d_weight = -2 * (X_train * (y_train - y_curr)).sum() / len(y_train)
            d_bias = -2 * (y_train - y_curr).sum() / len(y_train)
            self.weight -= d_weight * self.learning_rate
            self.bias -= d_bias * self.learning_rate
            self.score["MAE"].append(abs(y_train - y_curr).sum() / len(y_train))
            self.score["MAPE"].append(abs((y_train - y_curr) / y_train).sum() / len(y_train) * 100)
            self.score["MSE"].append(((y_train - y_curr)**2).sum() / len(y_train))
            self.score["RMSE"].append((((y_train - y_curr)**2).sum() / len(y_train)) ** (1/2))
            self.score["R2"].append((np.corrcoef(y_train, y_curr)[0, 1]) ** 2)
            if log == True:
                print("[Epoch %d] MSE Score : %f" %(i, self.score["MSE"][-1]))
        if lr_graph == True:
            plt.scatter(X_train, y_train)
            plt.plot(X_train, self.weight*X_train + self.bias, color = 'red')
            plt.title('OLS Single Linear Regression')
            plt.show()

    def get_score(self, method = "MSE"):
        if method == "MSE":
            print("MSE Score :", self.score["MSE"][-1])
        elif method == "MAE":
            print("MAE Score :", self.score["MAE"][-1])
        elif method == "MAPE":
            print("MAPE Score :", self.score["MAPE"][-1])
        elif method == "RMSE":
            print("RMSE Score :", self.score["RMSE"][-1])
        elif method == "R2":
            print("R2 Score :", self.score["R2"][-1])
        else:
            print("Not Supported")
            print("Support only MSE, MAE, MAPE, RMSE, R2")

    def predict(self, X_test):
        y_pred = self.weight * X_test + self.bias
        return y_pred

class GDMultiLinearRegression:

    def __init__(self, learning_rate = 0.01, epochs = 1000):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.col = 0
        self.weight = None
        self.bias = 0
        self.score = {"MAE": [], "MAPE": [], "MSE": [], "RMSE": [], "R2": []}

    def fit(self, X_train, y_train, log = False):
        X_train = np.array(X_train)
        self.col = X_train.shape[1]
        self.weight = [0] * self.col
        for i in range(self.epochs):
            y_curr = np.dot(self.weight, X_train.T) + self.bias
            d_weight = -2 * np.dot((y_train - y_curr).T, X_train) / len(y_train)
            d_bias = -2 * (y_train - y_curr).sum() / len(y_train)
            self.weight -= np.dot(d_weight, self.learning_rate)
            self.bias -= d_bias * self.learning_rate
            self.score["MAE"].append(abs(y_train - y_curr).sum() / len(y_train))
            self.score["MAPE"].append(abs((y_train - y_curr) / y_train).sum() / len(y_train) * 100)
            self.score["MSE"].append(((y_train - y_curr)**2).sum() / len(y_train))
            self.score["RMSE"].append((((y_train - y_curr)**2).sum() / len(y_train)) ** (1/2))
            self.score["R2"].append((np.corrcoef(y_train, y_curr)[0, 1]) ** 2)
            if log == True:
                print("[Epoch %d] MSE Score : %f" %(i, self.score["MSE"][-1]))

    def get_score(self, method = "MSE"):
        if method == "MSE":
            print("MSE Score :", self.score["MSE"][-1])
        elif method == "MAE":
            print("MAE Score :", self.score["MAE"][-1])
        elif method == "MAPE":
            print("MAPE Score :", self.score["MAPE"][-1])
        elif method == "RMSE":
            print("RMSE Score :", self.score["RMSE"][-1])
        elif method == "R2":
            print("R2 Score :", self.score["R2"][-1])
        else:
            print("Not Supported")
            print("Support only MSE, MAE, MAPE, RMSE, R2")

    def predict(self, X_test):
        y_pred = np.dot(self.weight, X_test.T) + self.bias
        return y_pred