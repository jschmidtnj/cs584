#!/usr/bin/env python3
"""
logistic regression
"""

import math
import numpy as np
from loguru import logger
from scipy.optimize import minimize
from sklearn.utils.extmath import safe_sparse_dot
from scipy.special import logsumexp
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder, LabelBinarizer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, roc_auc_score, roc_curve
import matplotlib.pyplot as plt

NUM_EPOCHS = 10
BATCH_SIZE = 32


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


class LogisticRegression:
    def __init__(self, learning_rate=0.1, tolerance=1e-4, max_iter=1000):
        # gradient descent parameters
        self.learning_rate = float(learning_rate)
        self.tolerance = float(tolerance)
        self.max_iter = int(max_iter)

        # how to construct a the design matrix
        self.add_intercept = True
        self.center = True
        self.scale = True

        self.training_loss_history = []

    def _design_matrix(self, X):
        if self.center:
            X = X - self.means
        if self.scale:
            X = X / self.standard_error
        if self.add_intercept:
            X = np.hstack([np.ones((X.shape[0], 1)), X])

        return X

    def fit_center_scale(self, X):
        self.means = X.mean(axis=0)
        self.standard_error = np.std(X, axis=0)

    def fit(self, X, y):
        self.fit_center_scale(X)

        # add intercept column to the design matrix
        n, k = X.shape
        X = self._design_matrix(X)

        # used for the convergence check
        previous_loss = -float('inf')
        self.converged = False

        # initialize parameters
        self.beta = np.zeros(k + (1 if self.add_intercept else 0))

        for i in range(self.max_iter):
            y_hat = sigmoid(X @ self.beta)
            self.loss = np.mean(-y * np.log(y_hat) - (1-y) * np.log(1-y_hat))

            # convergence check
            if abs(previous_loss - self.loss) < self.tolerance:
                self.converged = True
                break
            else:
                previous_loss = self.loss

            # gradient descent
            residuals = (y_hat - y).reshape((n, 1))
            gradient = (X * residuals).mean(axis=0)
            self.beta -= self.learning_rate * gradient

        self.iterations = i+1

    def predict_proba(self, X):
        # add intercept column to the design matrix
        X = self._design_matrix(X)
        return sigmoid(X @ self.beta)

    def predict(self, X):
        return (self.predict_proba(X) > 0.5).astype(int)

    def score(self, X, y, sample_weight=None):
        """
        Return the mean accuracy on the given test data and labels.

        In multi-label classification, this is the subset accuracy
        which is a harsh metric since you require for each sample that
        each label set be correctly predicted.
        """
        X = X.toarray()
        return accuracy_score(y, self.predict(X), sample_weight=sample_weight)


if __name__ == '__main__':
    raise RuntimeError("logistic regression cannot be run on its own")
