#!/usr/bin/env python3
"""
logistic regression
"""

import numpy as np
from loguru import logger
from scipy.optimize import minimize
from sklearn.utils.extmath import safe_sparse_dot
from scipy.special import logsumexp
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder, LabelBinarizer
from sklearn.linear_model import SGDClassifier
import math

NUM_EPOCHS = 10
BATCH_SIZE = 32

# SGD-based logistic regression, which has minibatch also

class LogisticRegression:
    """
    An L2-regularized linear model that uses SGD to minimize the in-sample error function.
    """
    
    def __init__(self, learning_rate=0.01, regularization_strength=0.0, **args):
        """
        Initialize the linear model.
        """
        self._w = None
        self._n_epochs = NUM_EPOCHS
        self._learning_rate = learning_rate
        self._batch_size = BATCH_SIZE
        self._regularization_strength = regularization_strength   
            
    def fit(self, X, y):
        """
        Fit the model with training data.
        """
        X = X.toarray()
        X = np.hstack((np.ones((X.shape[0], 1)), X)) # x_0 is always 1
        self._w = np.random.randn(X.shape[1], 1)
        batch_size = self._batch_size if self._batch_size is not None else X.shape[0]
        for i in range(self._n_epochs):
            print(i)
            for j in range(int(X.shape[0] / batch_size)):
                learning_rate = self._learning_rate if isinstance(self._learning_rate, float) \
                                else self._learning_rate(i * (X.shape[0] / batch_size) + j)
                sample = np.random.choice(X.shape[0], batch_size, replace=False)
                self._w -= learning_rate * self.gradient(X[sample,:], y[sample])

    def theta(self, s):
        return (math.e ** s) / (1 + math.e ** s)
    
    def gradient(self, X, y):
        gradient = np.zeros((X.shape[1], 1))
        for xi, yi in zip(X, y):
            gradient += np.reshape(self.theta(-yi * np.dot(np.transpose(self._w), xi)) * yi * xi,
                                   (X.shape[1], 1))
        gradient *= (-1.0 / X.shape[0])
        return gradient + 2.0 * self._regularization_strength * self._w
    
    def predict(self, X):
        X = np.hstack((np.ones((X.shape[0], 1)), X)) # x_0 is always 1
        res = np.vectorize(lambda x: self.theta(x))(
                np.dot(np.transpose(self._w), np.transpose(X)).flatten()
        )
        print(res)
        rounded = np.rint(res)
        print(rounded)
        return rounded

    def score(self, X, y, sample_weight=None):
        """
        Return the mean accuracy on the given test data and labels.
        """
        X = X.toarray()
        return accuracy_score(y, self.predict(X), sample_weight=sample_weight)


if __name__ == '__main__':
    raise RuntimeError("logistic regression cannot be run on its own")
