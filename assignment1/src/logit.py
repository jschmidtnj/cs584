#!/usr/bin/env python3
"""
logistic regression
"""

import numpy as np
from loguru import logger
from sklearn.metrics import accuracy_score

BATCH_SIZE = 32

# mini-batch logistic regression

class LogisticRegression:

    def _softmax(self, x):
        """
        Compute softmax
        """
        shape = x.shape
        num_rows = shape[0]
        output = np.zeros(shape)
        for row in range(num_rows):
            e_x = np.exp(x[row, :] - np.max(x[row, :]))
            output[row, :] = e_x / e_x.sum(axis=0)

        return output

    def _net(self, Xi):
        """
        Define out network and obtain a predicted output for a set of M inputs ( V > K ) 
        """
        for col in range(self.bias_matrix.shape[1]):
            self.bias_matrix[:, col] = self.bias_matrix[:, col].mean()

        y_linear = np.dot(Xi, self.weights) + self.bias_matrix[:Xi.shape[0], :]
        return self._softmax(y_linear)

    def __init__(self, **args):
        self.weights = None
        self.bias_matrix = None

    def score(self, X, y):
        return accuracy_score(y, self.predict(X))

    def predict(self, X):
        X = X.toarray()
        res = []
        for elem in X:
            net_output = self._net(elem)
            output = np.argmax(net_output[0])
            res.append(output)
        return res

    def _process_labels(self, y_train):
        res = []
        for val in y_train:
            res.append(np.zeros(3))
            res[-1][val] = 1
        return np.array(res)

    def fit(self, X_train_text, y_train):
        # training loop
        X_train_text = X_train_text.toarray()
        y_train = self._process_labels(y_train)
        N = X_train_text.shape[0]            # Dataset length
        V = X_train_text.shape[1]            # Vocab length
        K = 3                           # Number of classes
        self.weights = np.random.random((V, K))    # Weight vector
        b = np.random.random((1, K))    # Bias vector
        self.bias_matrix = np.repeat(b, BATCH_SIZE, axis=0)     # Bias matrix
        lr = 1e-6                       # Learning rate
        lmda = 1e-6                     # Regularization coefficient
        logger.success(f"Shape of N (Dataset length): {N}")
        logger.success(f"Shape of V (Vocab length): {V}")
        logger.success(f"Shape of M (Batch size): {BATCH_SIZE}")
        logger.success(f"Shape of K (Num classes): {K}")
        logger.success(f"Shape of W (Weight vector): {self.weights.shape}")
        logger.success(f"Shape of b (Bias): {b.shape}")
        logger.success(
            f"Shape of B (Bias matrix for iteration): {self.bias_matrix.shape}")

        epochs = 3
        dataset_length = X_train_text.shape[0]
        dataset_indexes = np.arange(dataset_length)
        batches = int(len(X_train_text) / BATCH_SIZE)
        for epoch in range(epochs):
            logger.info(f"Epoch: {epoch}")
            epoch_loss = 0
            epoch_gradient = np.zeros((BATCH_SIZE, K))
            for i in range(batches):
                for _ in range(BATCH_SIZE):
                    current_indices = np.random.choice(
                        dataset_indexes, BATCH_SIZE, replace=False)
                    Xi = X_train_text[current_indices]
                    Yi = y_train[current_indices]
                    yhat = self._net(Xi)
                    a_1 = -1/N * Yi * yhat
                    b_1 = lmda * (self.weights[current_indices, :]**2)
                    epoch_loss += a_1 + b_1
                    # logger.info(f"Epoch loss: {epoch_loss}")
                    # epoch_gradient += Yi * np.dot(1-net(Xi), Xi) + 2 * lmda * W[M, :]
                    epoch_gradient += np.add(Yi * (1-yhat),
                                             2 * lmda * self.weights[current_indices])
                    bias_epoch_gradient = (yhat - Yi)
                    # logger.info(f"Bias epoch gradient: {bias_epoch_gradient}")
                    # logger.info(f"Epoch gradient: {epoch_gradient}")
        self.weights[current_indices, :] = \
            self.weights[current_indices, :] - lr * epoch_gradient
        self.bias_matrix = self.bias_matrix - lr * bias_epoch_gradient
        logger.success("Weights updated")

        # end training loop
        logger.success("Completed Training Loop!")

        logger.info(f"Weights shape: {self.weights.shape}")
        logger.info(f"Bias shape: {self.bias_matrix.shape}")

        testX = X_train_text[0]
        testY = y_train[0]
        k = np.where(testY == 1)
        logger.info(f"testY: {testY}")
        logger.info(f"k = {k}")
        net_output = self._net(testX)
        logger.info(f"net output: {net_output[0]}")
        logger.info(f"net prediction: {np.argmax(net_output[0])}")
        return [], [], lmda


if __name__ == '__main__':
    raise RuntimeError("logistic regression cannot be run on its own")
