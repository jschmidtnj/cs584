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
from sklearn.metrics import confusion_matrix, accuracy_score, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import math
from sklearn.utils._weight_vector import WeightVector
from sklearn.linear_model._sgd_fast import Hinge

INFINITY = math.inf

# Penalty constants
NO_PENALTY = 0
L1 = 1
L2 = 2
ELASTICNET = 3

# Learning rate constants
CONSTANT = 1
OPTIMAL = 2
INVSCALING = 3
ADAPTIVE = 4
PA1 = 5
PA2 = 6

NUM_EPOCHS = 10
BATCH_SIZE = 32

# attempt to port to python failed

# pointers????
# does this work???
# https://towardsdatascience.com/implement-logistic-regression-with-l2-regularization-from-scratch-in-python-20bd4ee88a59#976b

# cdef class WeightVector(object):
#     """Dense vector represented by a scalar and a numpy array.
#     The class provides methods to ``add`` a sparse vector
#     and scale the vector.
#     Representing a vector explicitly as a scalar times a
#     vector allows for efficient scaling operations.
#     Attributes
#     ----------
#     w : ndarray, dtype=double, order='C'
#         The numpy array which backs the weight vector.
#     aw : ndarray, dtype=double, order='C'
#         The numpy array which backs the average_weight vector.
#     wscale : double
#         The scale of the vector.
#     n_features : int
#         The number of features (= dimensionality of ``w``).
#     sq_norm : double
#         The squared norm of ``w``.
#     """
#     def __cinit__(self, double [::1] w, double [::1] aw):
#         if w.shape[0] > INT_MAX:
#             raise ValueError("More than %d features not supported; got %d."
#                              % (INT_MAX, w.shape[0]))
#         self.wscale = 1.0
#         self.n_features = w.shape[0]
#         self.sq_norm = _dot(<int>w.shape[0], &w[0], 1, &w[0], 1)

#         self.w_data_ptr = &w[0]
#         if aw is not None:
#             self.aw_data_ptr = &aw[0]
#             self.average_a = 0.0
#             self.average_b = 1.0

#     cdef void add(self, double *x_data_ptr, int *x_ind_ptr, int xnnz,
#                   double c) nogil:
#         """Scales sample x by constant c and adds it to the weight vector.
#         This operation updates ``sq_norm``.
#         Parameters
#         ----------
#         x_data_ptr : double*
#             The array which holds the feature values of ``x``.
#         x_ind_ptr : np.intc*
#             The array which holds the feature indices of ``x``.
#         xnnz : int
#             The number of non-zero features of ``x``.
#         c : double
#             The scaling constant for the example.
#         """
#         cdef int j
#         cdef int idx
#         cdef double val
#         cdef double innerprod = 0.0
#         cdef double xsqnorm = 0.0

#         # the next two lines save a factor of 2!
#         cdef double wscale = self.wscale
#         cdef double* w_data_ptr = self.w_data_ptr

#         for j in range(xnnz):
#             idx = x_ind_ptr[j]
#             val = x_data_ptr[j]
#             innerprod += (w_data_ptr[idx] * val)
#             xsqnorm += (val * val)
#             w_data_ptr[idx] += val * (c / wscale)

#         self.sq_norm += (xsqnorm * c * c) + (2.0 * innerprod * wscale * c)

#     # Update the average weights according to the sparse trick defined
#     # here: https://research.microsoft.com/pubs/192769/tricks-2012.pdf
#     # by Leon Bottou
#     cdef void add_average(self, double *x_data_ptr, int *x_ind_ptr, int xnnz,
#                           double c, double num_iter) nogil:
#         """Updates the average weight vector.
#         Parameters
#         ----------
#         x_data_ptr : double*
#             The array which holds the feature values of ``x``.
#         x_ind_ptr : np.intc*
#             The array which holds the feature indices of ``x``.
#         xnnz : int
#             The number of non-zero features of ``x``.
#         c : double
#             The scaling constant for the example.
#         num_iter : double
#             The total number of iterations.
#         """
#         cdef int j
#         cdef int idx
#         cdef double val
#         cdef double mu = 1.0 / num_iter
#         cdef double average_a = self.average_a
#         cdef double wscale = self.wscale
#         cdef double* aw_data_ptr = self.aw_data_ptr

#         for j in range(xnnz):
#             idx = x_ind_ptr[j]
#             val = x_data_ptr[j]
#             aw_data_ptr[idx] += (self.average_a * val * (-c / wscale))

#         # Once the sample has been processed
#         # update the average_a and average_b
#         if num_iter > 1:
#             self.average_b /= (1.0 - mu)
#         self.average_a += mu * self.average_b * wscale

#     def dot(self, x_data_ptr, x_ind_ptr,
#                     xnnz):
#         """Computes the dot product of a sample x and the weight vector.
#         Parameters
#         ----------
#         x_data_ptr : double*
#             The array which holds the feature values of ``x``.
#         x_ind_ptr : np.intc*
#             The array which holds the feature indices of ``x``.
#         xnnz : int
#             The number of non-zero features of ``x`` (length of x_ind_ptr).
#         Returns
#         -------
#         innerprod : double
#             The inner product of ``x`` and ``w``.
#         """
#         innerprod = 0.0
#         w_data_ptr = self.w_data_ptr
#         for j in range(xnnz):
#             idx = x_ind_ptr[j]
#             innerprod += w_data_ptr[idx] * x_data_ptr[j]
#         innerprod *= self.wscale
#         return innerprod

#     def scale(self, c):
#         """Scales the weight vector by a constant ``c``.
#         It updates ``wscale`` and ``sq_norm``. If ``wscale`` gets too
#         small we call ``reset_swcale``."""
#         self.wscale *= c
#         self.sq_norm *= (c * c)
#         if self.wscale < 1e-9:
#             self.reset_wscale()

#     def reset_wscale(self):
#         """Scales each coef of ``w`` by ``wscale`` and resets it to 1. """
#         if self.aw_data_ptr != NULL:
#             _axpy(self.n_features, self.average_a,
#                   self.w_data_ptr, 1, self.aw_data_ptr, 1)
#             _scal(self.n_features, 1.0 / self.average_b, self.aw_data_ptr, 1)
#             self.average_a = 0.0
#             self.average_b = 1.0

#         _scal(self.n_features, self.wscale, self.w_data_ptr, 1)
#         self.wscale = 1.0

#     def norm(self):
#         """The L2 norm of the weight vector. """
#         return sqrt(self.sq_norm)

def sqnorm(x_data_ptr, x_ind_ptr, xnnz):
    x_norm = 0.0
    for j in range(xnnz):
        z = x_data_ptr[j]
        x_norm += z * z
    return x_norm

def _plain_sgd(weights,
               intercept,
               average_weights,
               average_intercept,
               loss,
               penalty_type,
               alpha, C,
               l1_ratio,
               dataset,
               validation_mask,
               early_stopping, validation_score_cb,
               n_iter_no_change,
               max_iter, tol, fit_intercept,
               verbose, shuffle, seed,
               weight_pos, weight_neg,
               learning_rate, eta0,
               power_t,
               t=1.0,
               intercept_decay=1.0,
               average=0):

    # get the data information into easy vars
    n_samples = dataset.n_samples
    n_features = weights.shape[0]

    w = WeightVector(weights, average_weights)
    w_ptr = weights[0]
    x_data_ptr = None
    x_ind_ptr = None
    ps_ptr = None

    # helper variables
    no_improvement_count = 0
    infinity = False
    xnnz = 0
    eta = 0.0
    p = 0.0
    update = 0.0
    sumloss = 0.0
    score = 0.0
    best_loss = INFINITY
    best_score = -INFINITY
    y = 0.0
    sample_weight = 0
    class_weight = 1.0
    count = 0
    epoch = 0
    i = 0
    is_hinge = isinstance(loss, Hinge)
    optimal_init = 0.0
    dloss = 0.0
    MAX_DLOSS = 1e12
    max_change = 0.0
    max_weight = 0.0

    sample_index = 0
    validation_mask_view = validation_mask

    # q vector is only used for L1 regularization
    q = None
    q_data_ptr = None
    if penalty_type == L1 or penalty_type == ELASTICNET:
        q = np.zeros((n_features,), dtype=np.float64, order="c")
        q_data_ptr = q.data
    u = 0.0

    if penalty_type == L2:
        l1_ratio = 0.0
    elif penalty_type == L1:
        l1_ratio = 1.0

    eta = eta0

    if learning_rate == OPTIMAL:
        typw = np.sqrt(1.0 / np.sqrt(alpha))
        # computing eta0, the initial learning rate
        initial_eta0 = typw / max(1.0, loss.dloss(-typw, 1.0))
        # initialize t such that eta at first sample equals eta0
        optimal_init = 1.0 / (initial_eta0 * alpha)

    for epoch in range(max_iter):
        sumloss = 0
        if verbose > 0:
            print("-- Epoch %d" % (epoch + 1))
        if shuffle:
            dataset.shuffle(seed)
        for i in range(n_samples):
            dataset.next(x_data_ptr, x_ind_ptr, xnnz,
                            y, sample_weight)

            sample_index = dataset.index_data_ptr[dataset.current_index]
            if validation_mask_view[sample_index]:
                # do not learn on the validation set
                continue

            p = w.dot(x_data_ptr, x_ind_ptr, xnnz) + intercept
            if learning_rate == OPTIMAL:
                eta = 1.0 / (alpha * (optimal_init + t - 1))
            elif learning_rate == INVSCALING:
                eta = eta0 / pow(t, power_t)

            if verbose or not early_stopping:
                sumloss += loss.loss(p, y)

            if y > 0.0:
                class_weight = weight_pos
            else:
                class_weight = weight_neg

            if learning_rate == PA1:
                update = sqnorm(x_data_ptr, x_ind_ptr, xnnz)
                if update == 0:
                    continue
                update = min(C, loss.loss(p, y) / update)
            elif learning_rate == PA2:
                update = sqnorm(x_data_ptr, x_ind_ptr, xnnz)
                update = loss.loss(p, y) / (update + 0.5 / C)
            else:
                dloss = loss.dloss(p, y)
                # clip dloss with large values to avoid numerical
                # instabilities
                if dloss < -MAX_DLOSS:
                    dloss = -MAX_DLOSS
                elif dloss > MAX_DLOSS:
                    dloss = MAX_DLOSS
                update = -eta * dloss

            if learning_rate >= PA1:
                if is_hinge:
                    # classification
                    update *= y
                elif y - p < 0:
                    # regression
                    update *= -1

            update *= class_weight * sample_weight

            if penalty_type >= L2:
                # do not scale to negative values when eta or alpha are too
                # big: instead set the weights to zero
                w.scale(max(0, 1.0 - ((1.0 - l1_ratio) * eta * alpha)))
            if update != 0.0:
                w.add(x_data_ptr, x_ind_ptr, xnnz, update)
                if fit_intercept == 1:
                    intercept += update * intercept_decay

            if 0 < average <= t:
                # compute the average for the intercept and update the
                # average weights, this is done regardless as to whether
                # the update is 0

                w.add_average(x_data_ptr, x_ind_ptr, xnnz,
                                update, (t - average + 1))
                average_intercept += ((intercept - average_intercept) /
                                        (t - average + 1))

            t += 1
            count += 1

        # report epoch information
        if verbose > 0:
            print("Norm: %.2f, NNZs: %d, Bias: %.6f, T: %d, "
                    "Avg. loss: %f"
                    % (w.norm(), weights.nonzero()[0].shape[0],
                        intercept, count, sumloss / n_samples))
            # print("Total training time: %.2f seconds."
            #         % (time() - t_start))

        # floating-point under-/overflow check.
        # if (not skl_isfinite(intercept)
        #     or any_nonfinite(weights.data, n_features)):
        #     infinity = True
        #     break

        # evaluate the score on the validation set
        if early_stopping:
            score = validation_score_cb(weights, intercept)
            if tol > -INFINITY and score < best_score + tol:
                no_improvement_count += 1
            else:
                no_improvement_count = 0
            if score > best_score:
                best_score = score
        # or evaluate the loss on the training set
        else:
            if tol > -INFINITY and sumloss > best_loss - tol * n_samples:
                no_improvement_count += 1
            else:
                no_improvement_count = 0
            if sumloss < best_loss:
                best_loss = sumloss

        # if there is no improvement several times in a row
        if no_improvement_count >= n_iter_no_change:
            if learning_rate == ADAPTIVE and eta > 1e-6:
                eta = eta / 5
                no_improvement_count = 0
            else:
                break

    if infinity:
        raise ValueError(("Floating-point under-/overflow occurred at epoch"
                          " #%d. Scaling input data with StandardScaler or"
                          " MinMaxScaler might help.") % (epoch + 1))

    w.reset_wscale()

    return weights, intercept, average_weights, average_intercept, epoch + 1