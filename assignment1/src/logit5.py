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

NUM_EPOCHS = 10
BATCH_SIZE = 32

# pointers????
# does this work???
# https://towardsdatascience.com/implement-logistic-regression-with-l2-regularization-from-scratch-in-python-20bd4ee88a59#976b

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
    w_ptr = &weights[0]
    *x_data_ptr = NULL
    cdef int *x_ind_ptr = NULL
    cdef double* ps_ptr = NULL

    # helper variables
    cdef int no_improvement_count = 0
    cdef bint infinity = False
    cdef int xnnz
    cdef double eta = 0.0
    cdef double p = 0.0
    cdef double update = 0.0
    cdef double sumloss = 0.0
    cdef double score = 0.0
    cdef double best_loss = INFINITY
    cdef double best_score = -INFINITY
    cdef double y = 0.0
    cdef double sample_weight
    cdef double class_weight = 1.0
    cdef unsigned int count = 0
    cdef unsigned int epoch = 0
    cdef unsigned int i = 0
    cdef int is_hinge = isinstance(loss, Hinge)
    cdef double optimal_init = 0.0
    cdef double dloss = 0.0
    cdef double MAX_DLOSS = 1e12
    cdef double max_change = 0.0
    cdef double max_weight = 0.0

    cdef long long sample_index
    cdef unsigned char [:] validation_mask_view = validation_mask

    # q vector is only used for L1 regularization
    cdef np.ndarray[double, ndim = 1, mode = "c"] q = None
    cdef double * q_data_ptr = NULL
    if penalty_type == L1 or penalty_type == ELASTICNET:
        q = np.zeros((n_features,), dtype=np.float64, order="c")
        q_data_ptr = <double * > q.data
    cdef double u = 0.0

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

    t_start = time()
    with nogil:
        for epoch in range(max_iter):
            sumloss = 0
            if verbose > 0:
                with gil:
                    print("-- Epoch %d" % (epoch + 1))
            if shuffle:
                dataset.shuffle(seed)
            for i in range(n_samples):
                dataset.next(&x_data_ptr, &x_ind_ptr, &xnnz,
                             &y, &sample_weight)

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

                if penalty_type == L1 or penalty_type == ELASTICNET:
                    u += (l1_ratio * eta * alpha)
                    l1penalty(w, q_data_ptr, x_ind_ptr, xnnz, u)

                t += 1
                count += 1

            # report epoch information
            if verbose > 0:
                with gil:
                    print("Norm: %.2f, NNZs: %d, Bias: %.6f, T: %d, "
                          "Avg. loss: %f"
                          % (w.norm(), weights.nonzero()[0].shape[0],
                             intercept, count, sumloss / n_samples))
                    print("Total training time: %.2f seconds."
                          % (time() - t_start))

            # floating-point under-/overflow check.
            if (not skl_isfinite(intercept)
                or any_nonfinite(<double *>weights.data, n_features)):
                infinity = True
                break

            # evaluate the score on the validation set
            if early_stopping:
                with gil:
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
                    if verbose:
                        with gil:
                            print("Convergence after %d epochs took %.2f "
                                  "seconds" % (epoch + 1, time() - t_start))
                    break

    if infinity:
        raise ValueError(("Floating-point under-/overflow occurred at epoch"
                          " #%d. Scaling input data with StandardScaler or"
                          " MinMaxScaler might help.") % (epoch + 1))

    w.reset_wscale()

    return weights, intercept, average_weights, average_intercept, epoch + 1