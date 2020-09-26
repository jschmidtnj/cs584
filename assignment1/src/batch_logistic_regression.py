#!/usr/bin/env python3
"""
batch logistic regression
"""

import numpy as np
from sklearn.utils.extmath import safe_sparse_dot
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder, LabelBinarizer
from scipy.optimize import minimize
from scipy.special import logsumexp


def _squared_norm(data):
    """Squared Euclidean norm of x. equivalent to norm(x) ** 2."""
    data = np.ravel(data, order='K')
    return np.dot(data, data)


def _get_loss(weights, X, Y, lmbda, sample_weight):
    """Computes the loss."""
    num_classes = Y.shape[1]
    num_features = X.shape[1]
    fit_intercept = weights.size == (num_classes * (num_features + 1))
    weights = weights.reshape(num_classes, -1)
    sample_weight = sample_weight[:, np.newaxis]
    if fit_intercept:
        intercept = weights[:, -1]
        weights = weights[:, :-1]
    else:
        intercept = 0
    prob = safe_sparse_dot(X, weights.T)
    prob += intercept
    prob -= logsumexp(prob, axis=1)[:, np.newaxis]
    loss = -(sample_weight * Y * prob).sum()
    loss += 0.5 * lmbda * _squared_norm(weights)
    prob = np.exp(prob, prob)
    return loss, prob, weights


def _loss_gradient(weights, X, Y, lmbda, sample_weight):
    """Computes loss and class probabilities."""
    num_classes = Y.shape[1]
    num_features = X.shape[1]
    fit_intercept = (weights.size == num_classes * (num_features + 1))
    gradient = np.zeros((num_classes, num_features + bool(fit_intercept)),
                        dtype=X.dtype)
    loss, prob, weights = _get_loss(weights, X, Y, lmbda, sample_weight)
    sample_weight = sample_weight[:, np.newaxis]
    diff = sample_weight * (prob - Y)
    gradient[:, :num_features] = safe_sparse_dot(diff.T, X)
    gradient[:, :num_features] += lmbda * weights
    if fit_intercept:
        gradient[:, -1] = diff.sum(axis=0)
    return loss, gradient.ravel(), prob


def _run_logistic_regression(X, y, lmbda=1, fit_intercept=True,
                             max_iter=100, tol=1e-4):
    """
    main logistic regression training function
    """

    _, num_features = X.shape

    classes = np.unique(y)

    sample_weight = np.ones(X.shape[0])

    label_encoder = LabelEncoder()
    class_weight = np.ones(len(classes))
    sample_weight *= class_weight[label_encoder.fit_transform(y)]

    lbin = LabelBinarizer()
    Y_multi = lbin.fit_transform(y)
    if Y_multi.shape[1] == 1:
        Y_multi = np.hstack([1 - Y_multi, Y_multi])

    w0 = np.zeros((classes.size, num_features + int(fit_intercept)),
                  order='F', dtype=X.dtype)

    w0 = w0.ravel()
    target = Y_multi

    training_losses = []
    validation_losses = []

    def callback(weights, X, Y, lmbda, sample_weights):
        res = _loss_gradient(weights, X, Y, lmbda, sample_weights)[0:2]
        training_losses.append(res[0])
        validation_losses.append(res[0])
        return res

    coefs = []

    # minimize
    opt_res = minimize(
        callback, w0, method="l-bfgs-b", jac=True,
        args=(X, target, 1. / lmbda, sample_weight),
        options={"gtol": tol, "maxiter": max_iter}
    )
    w0 = opt_res.x

    num_classes = max(2, classes.size)
    multi_w0 = np.reshape(w0, (num_classes, -1))
    if num_classes == 2:
        multi_w0 = multi_w0[1][np.newaxis, :]
    coefs.append(multi_w0.copy())

    return np.array(coefs), np.array([lmbda]), training_losses, validation_losses


class BatchLogisticRegression:
    """
    batch logistic regression
    uses minimization function. is different from mini-batch
    """

    def __init__(self, tol=1e-4, fit_intercept=True,
                 max_iter=100, lmbda=1, **_args):

        self.tol = tol
        self.fit_intercept = fit_intercept
        self.max_iter = max_iter
        self.lmbda = lmbda
        self.training_losses = None
        self.validation_losses = None
        self.classes = None
        self.coefficients = []
        self.intercept = None

    def fit(self, X, y):
        """
        batch logistic regression fit function (runs training)
        """

        self.classes = np.unique(y)

        num_classes = len(self.classes)

        self.coefficients = []
        self.intercept = np.zeros(num_classes)

        res = [_run_logistic_regression(X, y, lmbda=self.lmbda, fit_intercept=self.fit_intercept,
                                        tol=self.tol, max_iter=self.max_iter)]

        fold_coefficients, _, training_losses, validation_losses = zip(
            *res)

        self.coefficients = fold_coefficients[0][0]
        self.intercept = self.coefficients[:, -1]
        self.coefficients = self.coefficients[:, :-1]
        self.training_losses = training_losses[0]
        self.validation_losses = validation_losses[0]

    def get_train_validation_loss(self):
        """
        get training loss and validation loss data
        """
        return self.training_losses, self.validation_losses

    def _main_decision_func(self, X):
        """
        Predict confidence scores for samples.
        """
        scores = safe_sparse_dot(X, self.coefficients.T,
                                 dense_output=True) + self.intercept
        return scores.ravel() if scores.shape[1] == 1 else scores

    def predict(self, X):
        """
        Predict class labels for samples in X.
        """
        scores = self._main_decision_func(X)
        if len(scores.shape) == 1:
            indices = (scores > 0).astype(np.int)
        else:
            indices = scores.argmax(axis=1)
        res = self.classes[indices]
        return res

    def score(self, X, y, sample_weight=None):
        """
        Return the mean accuracy on the given test data and labels.
        """
        return accuracy_score(y, self.predict(X), sample_weight=sample_weight)


if __name__ == '__main__':
    raise RuntimeError("logistic regression cannot be run on its own")
