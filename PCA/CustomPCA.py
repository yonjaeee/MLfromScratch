import numpy as np
import matplotlib.pyplot as plt

class PCA:

    def __init__(self, n_components=None):
        self.n_components = n_components

    def fit(self, X):
        # Normalize X
        X_norm = X - X.mean(axis = 0)
        X_norm = X_norm / X.std(axis = 0)
        # Covariance Matrix
        cov_matrix = np.cov(X_norm, rowvar = False)
        # Eigendecomposition
        eigen_val, eigen_vec = np.linalg.eigh(cov_matrix)
        # Sort Eigen Value, Eigen Vector in Descending Order
        sorted_idx = np.argsort(eigen_val)[::-1]
        sorted_eigen_val = eigen_val[sorted_idx]
        sorted_eigen_vec = eigen_vec[:, sorted_idx]
        # Select n_components
        if self.n_components != None:
            eigen_vec_sel = sorted_eigen_vec[:, :self.n_components]
            self.eigen_selected = eigen_vec_sel
        else:
            eigen_vec_sel = sorted_eigen_vec
            self.eigen_selected = eigen_vec_sel
        self.explained_var_ratio = self.eigen_selected / sorted_eigen_val.sum()
        # Project X into Principal Component Area
        X_pca = X.dot(eigen_vec_sel)
        return X_pca

    def plot_evr(self):
        x = ['pca_'+ str(x) for x in range(1, self.n_components + 1)]
        plt.bar(x, self.explained_var_ratio)
        plt.title('Explained Variance Ratio')
        plt.ylabel('Ratio')
        plt.show()
