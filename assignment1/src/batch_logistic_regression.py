#!/usr/bin/env python3
"""
logistic regression
"""

import numpy as np
from scipy.optimize import minimize
from sklearn.utils.extmath import safe_sparse_dot
from scipy.special import logsumexp
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder, LabelBinarizer

def squared_norm(x):
    """Squared Euclidean norm of x. equivalent to norm(x) ** 2."""
    x = np.ravel(x, order='K')
    return np.dot(x, x)


def get_loss(w, X, Y, alpha, sample_weight):
    """Computes the loss."""
    num_classes = Y.shape[1]
    num_features = X.shape[1]
    fit_intercept = w.size == (num_classes * (num_features + 1))
    w = w.reshape(num_classes, -1)
    sample_weight = sample_weight[:, np.newaxis]
    if fit_intercept:
        intercept = w[:, -1]
        w = w[:, :-1]
    else:
        intercept = 0
    p = safe_sparse_dot(X, w.T)
    p += intercept
    p -= logsumexp(p, axis=1)[:, np.newaxis]
    loss = -(sample_weight * Y * p).sum()
    loss += 0.5 * alpha * squared_norm(w)
    p = np.exp(p, p)
    return loss, p, w


def loss_grad(w, X, Y, alpha, sample_weight):
    """Computes loss and class probabilities."""
    num_classes = Y.shape[1]
    num_features = X.shape[1]
    fit_intercept = (w.size == num_classes * (num_features + 1))
    grad = np.zeros((num_classes, num_features + bool(fit_intercept)),
                    dtype=X.dtype)
    loss, p, w = get_loss(w, X, Y, alpha, sample_weight)
    sample_weight = sample_weight[:, np.newaxis]
    diff = sample_weight * (p - Y)
    grad[:, :num_features] = safe_sparse_dot(diff.T, X)
    grad[:, :num_features] += alpha * w
    if fit_intercept:
        grad[:, -1] = diff.sum(axis=0)
    return loss, grad.ravel(), p


def _logistic_regression(X, y, Cs=10, fit_intercept=True,
                              max_iter=100, tol=1e-4):

    _, num_features = X.shape

    classes = np.unique(y)

    sample_weight = np.ones(X.shape[0])

    le = LabelEncoder()
    class_weight_ = np.ones(len(classes))
    sample_weight *= class_weight_[le.fit_transform(y)]

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
    last_lambda = [-1]

    def callback(w, X, Y, alpha, sample_weight):
        res = loss_grad(w, X, Y, alpha, sample_weight)[0:2]
        training_losses.append(res[0])
        validation_losses.append(res[0])
        # alpha is lambda
        last_lambda[0] = alpha
        return res

    coefs = []
    for i, C in enumerate(Cs):

        # # TODO - add for loop with batch size for minibatch
        # for _ in range(max_iter):
        #     grad_res = loss_grad(w0, X, target, 1. / C, sample_weight)
        #     # print(grad_res[0], grad_res[2])
        #     w0 -= .1 * grad_res[1]

        # # TODO - loop for sgd
        # batch_size = 32
        # for k in range(max_iter):
        #     print(k)
        #     for j in range(int(X.shape[0] / batch_size)):
        #         sample = np.random.choice(X.shape[0], batch_size, replace=False)
        #         grad_res = loss_grad(w0, X[sample], target[sample], 1. / C, sample_weight[sample])
        #         w0 -= .05 * grad_res[1]

        # loop for batch
        opt_res = minimize(
            callback, w0, method="l-bfgs-b", jac=True,
            args=(X, target, 1. / C, sample_weight),
            options={"gtol": tol, "maxiter": max_iter}
        )
        w0 = opt_res.x

        num_classes = max(2, classes.size)
        multi_w0 = np.reshape(w0, (num_classes, -1))
        if num_classes == 2:
            multi_w0 = multi_w0[1][np.newaxis, :]
        coefs.append(multi_w0.copy())

    return np.array(coefs), np.array(Cs), training_losses, validation_losses, last_lambda


class LogisticRegression:
    def __init__(self, tol=1e-4, fit_intercept=True,
                 max_iter=100, Cs=[1], **_args):

        self.tol = tol
        self.fit_intercept = fit_intercept
        self.max_iter = max_iter
        self.Cs = Cs

    def fit(self, X, y):

        self.classes_ = np.unique(y)

        num_classes = len(self.classes_)

        self.coefficients = []
        self.intercept_ = np.zeros(num_classes)

        res = [_logistic_regression(X, y, Cs=self.Cs, fit_intercept=self.fit_intercept,
                                                         tol=self.tol,
                                                         max_iter=self.max_iter)]

        fold_coefficients, _, training_losses, validation_losses, last_lambda = zip(*res)
        training_losses = training_losses[0]
        validation_losses = validation_losses[0]
        last_lambda = last_lambda[0][0]

        self.coefficients = fold_coefficients[0][0]
        self.intercept_ = self.coefficients[:, -1]
        self.coefficients = self.coefficients[:, :-1]
        return training_losses, validation_losses, last_lambda

    def decision_function(self, X):
        """
        Predict confidence scores for samples.
        """
        scores = safe_sparse_dot(X, self.coefficients.T,
                                 dense_output=True) + self.intercept_
        return scores.ravel() if scores.shape[1] == 1 else scores

    def predict(self, X):
        """
        Predict class labels for samples in X.
        """
        scores = self.decision_function(X)
        if len(scores.shape) == 1:
            indices = (scores > 0).astype(np.int)
        else:
            indices = scores.argmax(axis=1)
        res = self.classes_[indices]
        return res

    def score(self, X, y, sample_weight=None):
        """
        Return the mean accuracy on the given test data and labels.
        """
        return accuracy_score(y, self.predict(X), sample_weight=sample_weight)


if __name__ == '__main__':
    raise RuntimeError("logistic regression cannot be run on its own")
