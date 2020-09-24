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

BATCH_SIZE = 32

# https://machinelearningmastery.com/implement-logistic-regression-stochastic-gradient-descent-scratch-python/
# https://github.com/iamkucuk/Logistic-Regression-With-Mini-Batch-Gradient-Descent/blob/master/logistic_regression_notebook.ipynb
# https://www.geeksforgeeks.org/ml-mini-batch-gradient-descent-with-python/
# http://www.oranlooney.com/post/ml-from-scratch-part-2-logistic-regression/
# https://stats.stackexchange.com/a/117928 - mini-batch vs batch vs epoch
# https://towardsdatascience.com/understanding-the-scaling-of-l%C2%B2-regularization-in-the-context-of-neural-networks-e3d25f8b50db
# https://github.com/sergei-bondarenko/machine-learning/blob/master/l2.ipynb
# https://github.com/ral99/SGDForLinearModels/blob/master/pysgd/linear_models.py

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


def _logistic_regression_path(X, y, Cs=10, fit_intercept=True,
                              max_iter=100, tol=1e-4, verbose=0,
                              pos_class=None, coef=None):

    # Preprocessing.
    _, n_features = X.shape
    print(X.shape, y.shape)

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

    w0 = w0.ravel()
    target = Y_multi

    def func(w, X, Y, alpha, sample_weight):
        res = _multinomial_loss_grad(w, X, Y, alpha, sample_weight)[0:2]
        # print(res[0], res[1])
        loss = res[0]
        # TODO - compute the score here...
        # alpha is lambda
        # print(loss)
        # print(alpha, loss)
        return res

    coefs = list()
    n_iter = np.zeros(len(Cs), dtype=np.int32)
    for i, C in enumerate(Cs):
        # TODO - add for loop with batch size for minibatch
        # for _ in range(max_iter):
        #     grad_res = _multinomial_loss_grad(w0, X, target, 1. / C, sample_weight)
        #     # print(grad_res[0], grad_res[2])
        #     w0 -= .1 * grad_res[1]
        # n_iter_i = max_iter
        num_epoch = 20
        for k in range(num_epoch):
            print(k)
            for j in range(int(X.shape[0] / BATCH_SIZE)):
                sample = np.random.choice(X.shape[0], BATCH_SIZE, replace=False)
                grad_res = _multinomial_loss_grad(w0, X[sample], target[sample], 1. / C, sample_weight[sample])
                w0 -= .05 * grad_res[1] # self.gradient(X[sample,:], target[sample])
        n_iter_i = num_epoch
        # opt_res = minimize(
        #     func, w0, method="l-bfgs-b", jac=True,
        #     args=(X, target, 1. / C, sample_weight),
        #     options={"gtol": tol, "maxiter": max_iter}
        # )
        # n_iter_i = min(opt_res.nit, max_iter)
        # w0, loss = opt_res.x, opt_res.fun
        # # print('LAST!!!')
        # # print(loss, w0)

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
        C_s = [1]
        penalty = self.penalty

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
            fold_coefs_.append(_logistic_regression_path(X, y, Cs=C_s, fit_intercept=self.fit_intercept,
                                                         tol=self.tol, verbose=self.verbose,
                                                         max_iter=self.max_iter, coef=warm_start_coef_,
                                                         pos_class=class_))

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
        res = self.classes_[indices]
        print('output', res)
        return res

    def score(self, X, y, sample_weight=None):
        """
        Return the mean accuracy on the given test data and labels.

        In multi-label classification, this is the subset accuracy
        which is a harsh metric since you require for each sample that
        each label set be correctly predicted.
        """
        return accuracy_score(y, self.predict(X), sample_weight=sample_weight)


if __name__ == '__main__':
    raise RuntimeError("logistic regression cannot be run on its own")
