#!/usr/bin/env python3
"""
logistic regression
"""

import numpy as np
from loguru import logger
from scipy import optimize
from sklearn.utils.extmath import safe_sparse_dot
from scipy.special import logsumexp
from sklearn.preprocessing import LabelEncoder, LabelBinarizer

BATCH_SIZE = 5

def squared_norm(x):
    """Squared Euclidean or Frobenius norm of x.

    Faster than norm(x) ** 2.
    """
    x = np.ravel(x, order='K')
    return np.dot(x, x)

def _multinomial_loss(w, X, Y, alpha, sample_weight):
    """Computes the multinomial loss, gradient and class probabilities."""
    n_classes = Y.shape[1]
    n_features = X.shape[1]
    fit_intercept = w.size == (n_classes * (n_features + 1))
    w = w.reshape(n_classes, -1)
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

def _multinomial_loss_grad(w, X, Y, alpha, sample_weight):
    """Computes multinomial loss and class probabilities."""
    n_classes = Y.shape[1]
    n_features = X.shape[1]
    fit_intercept = (w.size == n_classes * (n_features + 1))
    grad = np.zeros((n_classes, n_features + bool(fit_intercept)),
                    dtype=X.dtype)
    loss, p, w = _multinomial_loss(w, X, Y, alpha, sample_weight)
    sample_weight = sample_weight[:, np.newaxis]
    diff = sample_weight * (p - Y)
    grad[:, :n_features] = safe_sparse_dot(diff.T, X)
    grad[:, :n_features] += alpha * w
    if fit_intercept:
        grad[:, -1] = diff.sum(axis=0)
    return loss, grad.ravel(), p

def _logistic_regression_path(X, y, pos_class=None, Cs=10, fit_intercept=True,
                              max_iter=100, tol=1e-4, verbose=0,
                              solver='lbfgs', coef=None,
                              class_weight=None, dual=False, penalty='l2',
                              intercept_scaling=1.,
                              random_state=None, check_input=True,
                              max_squared_sum=None, l1_ratio=None):

    # Preprocessing.
    _, n_features = X.shape

    classes = np.unique(y)

    # If sample weights exist, convert them to array (support for lists)
    # and check length
    # Otherwise set them to 1 for all examples
    sample_weight = np.ones(X.shape[0])

    # If class_weights is a dict (provided by the user), the weights
    # are assigned to the original labels. If it is "balanced", then
    # the class_weights are assigned after masking the labels with a OvR.
    le = LabelEncoder()
    class_weight_ = np.ones(len(classes))
    sample_weight *= class_weight_[le.fit_transform(y)]

    lbin = LabelBinarizer()
    Y_multi = lbin.fit_transform(y)
    if Y_multi.shape[1] == 1:
        Y_multi = np.hstack([1 - Y_multi, Y_multi])

    w0 = np.zeros((classes.size, n_features + int(fit_intercept)),
                    order='F', dtype=X.dtype)

    if coef is not None:
        # For binary problems coef.shape[0] should be 1, otherwise it
        # should be classes.size.
        n_classes = classes.size
        if n_classes == 2:
            n_classes = 1
        w0[:, :coef.shape[1]] = coef

    # scipy.optimize.minimize and newton-cg accepts only
    # ravelled parameters.
    w0 = w0.ravel()
    target = Y_multi
    def func(x, *args):
        return _multinomial_loss_grad(x, *args)[0:2]

    coefs = list()
    n_iter = np.zeros(len(Cs), dtype=np.int32)
    for i, C in enumerate(Cs):
        iprint = [-1, 50, 1, 100, 101][
            np.searchsorted(np.array([0, 1, 2, 3]), verbose)]
        opt_res = optimize.minimize(
            func, w0, method="L-BFGS-B", jac=True,
            args=(X, target, 1. / C, sample_weight),
            options={"iprint": iprint, "gtol": tol, "maxiter": max_iter}
        )
        n_iter_i = min(opt_res.nit, max_iter)
        w0, loss = opt_res.x, opt_res.fun

        n_classes = max(2, classes.size)
        multi_w0 = np.reshape(w0, (n_classes, -1))
        if n_classes == 2:
            multi_w0 = multi_w0[1][np.newaxis, :]
        coefs.append(multi_w0.copy())

        n_iter[i] = n_iter_i

    return np.array(coefs), np.array(Cs), n_iter

class LogisticRegression:
    def __init__(self, penalty='l2', *, dual=False, tol=1e-4, C=1.0,
                 fit_intercept=True, intercept_scaling=1, class_weight=None,
                 random_state=None, solver='lbfgs', max_iter=100,
                 verbose=0, warm_start=False, n_jobs=None,
                 l1_ratio=None):

        self.penalty = penalty
        self.dual = dual
        self.tol = tol
        self.C = C
        self.fit_intercept = fit_intercept
        self.intercept_scaling = intercept_scaling
        self.class_weight = class_weight
        self.random_state = random_state
        self.solver = solver
        self.max_iter = max_iter
        self.verbose = verbose
        self.warm_start = warm_start
        self.n_jobs = n_jobs
        self.l1_ratio = l1_ratio

    def fit(self, X, y):
        solver = self.solver
        C_ = self.C
        penalty = self.penalty

        _dtype = np.float64

        self.classes_ = np.unique(y)

        max_squared_sum = None

        n_classes = len(self.classes_)
        classes_ = self.classes_

        warm_start_coef = None

        self.coef_ = list()
        self.intercept_ = np.zeros(n_classes)

        # Hack so that we iterate only once for the multinomial case.
        classes_ = [None]
        warm_start_coef = [warm_start_coef]

        fold_coefs_ = []
        for class_, warm_start_coef_ in zip(classes_, warm_start_coef):
            fold_coefs_.append(_logistic_regression_path(X, y, pos_class=class_, Cs=[C_],
                      l1_ratio=self.l1_ratio, fit_intercept=self.fit_intercept,
                      tol=self.tol, verbose=self.verbose, solver=solver,
                      max_iter=self.max_iter,
                      class_weight=self.class_weight, check_input=False,
                      random_state=self.random_state, coef=warm_start_coef_,
                      penalty=penalty, max_squared_sum=max_squared_sum))

        fold_coefs_, _, n_iter_ = zip(*fold_coefs_)
        self.n_iter_ = np.asarray(n_iter_, dtype=np.int32)[:, 0]

        n_features = X.shape[1]
        self.coef_ = fold_coefs_[0][0]
        self.intercept_ = self.coef_[:, -1]
        self.coef_ = self.coef_[:, :-1]

    def decision_function(self, X):
        """
        Predict confidence scores for samples.
        """

        n_features = self.coef_.shape[1]
        if X.shape[1] != n_features:
            raise ValueError("X has %d features per sample; expecting %d"
                             % (X.shape[1], n_features))

        scores = safe_sparse_dot(X, self.coef_.T,
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
        return self.classes_[indices]

    def score(self, X, y, sample_weight=None):
        """
        Return the mean accuracy on the given test data and labels.

        In multi-label classification, this is the subset accuracy
        which is a harsh metric since you require for each sample that
        each label set be correctly predicted.
        """
        from sklearn.metrics import accuracy_score
        return accuracy_score(y, self.predict(X), sample_weight=sample_weight)


if __name__ == '__main__':
    raise RuntimeError("logistic regression cannot be run on its own")
