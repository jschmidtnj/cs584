#!/usr/bin/env python3
"""
minibatch logistic regression
"""

import numpy as np
# from loguru import logger
from sklearn.metrics import accuracy_score

# relative imports
from books import start_end_map


def softmax(x, multi=True):
    """
    get the softmax
    """
    ex = np.exp(x - np.max(x))
    return ex / ex.sum(axis=1 if multi else 0)


def _process_labels(y_train):
    """
    process labels to array
    """
    res = []
    for val in y_train:
        res.append(np.zeros(3))
        res[-1][val] = 1
    return np.array(res)


class LogisticRegression:
    """
    logistic regression
    """

    def __init__(self, epochs=100, batch_size=32, lmbda=1e-4, plot_epoch_iter=1, **_args):
        """
        logistic regression init
        """
        self.epochs = epochs
        self.weights = None
        self.bias_matrix = None
        self.batch_size = batch_size
        self.lmbda = lmbda
        self.bias_vector = None
        self.plot_epoch_iter = plot_epoch_iter
        self.training_scores = None
        self.testing_scores = None

    def _regularization(self, index, k):
        return self.lmbda * (self.weights[index, k]**2)

    def _regularization_gradient(self, k):
        return 2 * self.lmbda * self.weights[:, k]

    def _net(self, Xi, multi=True):
        """
        Define out network and obtain a predicted output for a set of M inputs ( V > K )
        """
        y_linear = np.add(np.dot(Xi, self.weights), self.bias_vector[0])
        return softmax(y_linear, multi=multi)

    def score(self, X, y):
        """
        returns the accuracy score of the logistic regression function
        """
        try:
            return accuracy_score(y, self.predict(X))
        except ValueError:
            return 0.

    def predict(self, X):
        """
        get prediction for given array of inputs
        """
        X = X.toarray()
        res = []
        for elem in X:
            net_output = self._net(elem, multi=False)
            output = np.argmax(net_output[0])
            res.append(output)
        return res

    def fit(self, X_train_text, y_train, X_test_text, y_test):
        """
        main training loop
        """
        X_train = X_train_text
        X_train_text = X_train_text.toarray()
        y_train = _process_labels(y_train)
        N = X_train_text.shape[0]  # dataset length
        V = X_train_text.shape[1]  # vocabulary length
        K = len(start_end_map.keys())  # num of classes
        lr = 1e-2  # Learning rate
        self.weights = np.random.rand(V, K)  # weight vector
        self.bias_vector = np.random.rand(1, K)  # bias vector
        self.bias_matrix = np.repeat(
            self.bias_vector, self.batch_size, axis=0)  # bias matrix

        dataset_len = X_train_text.shape[0]
        dataset_indexes = np.arange(dataset_len)
        training_scores = []
        testing_scores = []
        for epoch in range(self.epochs):
            # logger.info(f"Epoch: {epoch + 1}")
            epoch_loss = 0
            epoch_gradient = np.zeros((V, K))
            for _batch_index in range(0, N, self.batch_size):
                current_indices = np.random.choice(
                    dataset_indexes, self.batch_size, replace=False)
                X = X_train_text[current_indices]
                Y = y_train[current_indices]
                for index, Xi, Yi in zip(current_indices, X, Y):
                    Xi = Xi.reshape(1, V)
                    Yi = Yi.reshape(1, K)
                    yhat = self._net(Xi, multi=(
                        self.batch_size != 1)).reshape(1, K)
                    k_true = np.where(Yi[0] == 1)
                    # Prediction component of loss update
                    epoch_loss += -(1/N) * (np.dot(np.transpose(Yi), yhat))
                    # Prediction component of epoch gradient update
                    a = (np.transpose(Xi) @ (1-yhat))[:, k_true]
                    epoch_gradient[:,
                                   k_true] += a
                    # for all classes
                    for k in range(K):
                        # regularization update per class
                        epoch_loss += self._regularization(index, k)
                        epoch_gradient[:,
                                       k] += self._regularization_gradient(k)
            if (epoch + 1) % self.plot_epoch_iter == 0:
                training_scores.append(
                    self.score(X_train, y_train))
                testing_scores.append(
                    self.score(X_test_text, y_test))
            self.weights = self.weights - lr * epoch_gradient
        self.training_scores = training_scores
        self.testing_scores = testing_scores

    def get_train_test_scores(self):
        """
        get training and testing score data (for plotting)
        """
        return self.training_scores, self.testing_scores


if __name__ == '__main__':
    raise RuntimeError("logistic regression cannot be run on its own")
