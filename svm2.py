import numpy as np
import pandas as pd

class SVM:
    def __init__(self, n_iter=1000, lambda_param=0.01, alpha=0.01):
        """
        Initialize the SVM classifier.

        Parameters:
        - n_iter: Number of iterations for training (default: 1000)
        - lambda_param: Regularization parameter (default: 0.01)
        - alpha: Learning rate (default: 0.01)
        """
        self.n_iter = n_iter
        self.lambda_param = lambda_param
        self.alpha = alpha

    def fit(self, X, y):
        """
        Fit the SVM model to the training data.

        Parameters:
        - X: Input features, a 2D array-like object [n_samples, n_features]
        - y: Target values, a 1D array-like object [n_samples]

        Returns:
        None
        """
        # Validate input data
        X = np.array(X)
        y = np.array(y)
        self._validate_input(X, y)

        n_samples, n_features = X.shape

        self.w = np.zeros(n_features)
        self.b = 0

        y_ = np.where(y <= 0, -1, 1)

        for _ in range(self.n_iter):
            for idx, x_i in enumerate(X):
                condition = y_[idx] * (np.dot(x_i, self.w) + self.b) >= 1

                if condition:
                    dw = self.lambda_param * self.w
                    db = 0
                else:
                    dw = self.lambda_param * self.w - np.dot(x_i, y_[idx])
                    db = y_[idx]

                self.w -= self.alpha * dw
                self.b -= self.alpha * db

    def predict(self, X):
        """
        Predict the class labels for input data.

        Parameters:
        - X: Input features, a 2D array-like object [n_samples, n_features]

        Returns:
        - Predicted class labels, a 1D array-like object [n_samples]
        """
        X = np.array(X)
        self._validate_input(X)

        approx = np.dot(X, self.w) + self.b
        pred = np.sign(approx)
        return np.where(pred == -1, 0, 1)

    def _validate_input(self, X, y=None):
        """
        Validate the input data and target values.

        Parameters:
        - X: Input features, a 2D array-like object [n_samples, n_features]
        - y: Target values, a 1D array-like object [n_samples] (optional)

        Returns:
        None
        """
        if len(X.shape) != 2:
            raise ValueError("Input features must be a 2D array-like object.")

        if y is not None and len(y.shape) != 1:
            raise ValueError("Target values must be a 1D array-like object.")

        if y is not None and X.shape[0] != y.shape[0]:
            raise ValueError("Number of samples in X and y must match.")

        if not np.issubdtype(X.dtype, np.number):
            raise ValueError("Input features must be numeric.")

        if y is not None and not np.issubdtype(y.dtype, np.number):
            raise ValueError("Target values must be numeric.")


# Example usage
X_train = [[1, 2], [3, 4], [5, 6]]
y_train = [0, 1, 0]

X_test = [[2, 2], [4, 4]]

svm = SVM(n_iter=10000, lambda_param=0.1, alpha=0.001)
svm.fit(X_train, y_train)
predictions = svm.predict(X_test)
print(predictions)
