#!/usr/bin/env python3
"""
References:
    https://web.stanford.edu/~jurafsky/slp3/5.pdf
    https://ttic.uchicago.edu/~suriya/website-intromlss2018/course_material/Day3b.pdf
    https://gluon.mxnet.io/chapter02_supervised-learning/softmax-regression-scratch.html
    
"""

import re
import string
import pandas as pd
import numpy as np
from os import listdir
from typing import List, Dict
from loguru import logger
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer


def list_files(directory, extension) -> List[str]:
    """
    list files
    """
    return [f for f in listdir(directory) if f.endswith('.' + extension)]

def category_to_multiclass(n, num_classes):
    output = np.zeros(num_classes)
    # output = [0]*num_classes
    output[n] = 1

    return output

def load_train_test():
    data_path = "data"
    raw_data_fliles = list_files(data_path, "txt")
    table = str.maketrans('', '', string.punctuation)

    labels = []
    paragraphs = []

    for doc in raw_data_fliles:
        with open(f"{data_path}/{doc}", 'r') as f:
            contents = f.read()
            author = re.search(r'Author:\s[A-Z][a-z]+\s[A-Z][a-z]+', contents).group().replace("Author: ", "")

            contents = contents.split("\n\n")

            contents[:] = [x.translate(table) for x in contents if x]

            print(f"Length of data with label {author} : {len(contents)}")

            paragraphs += contents
            labels += [author]*len(contents)

    return paragraphs, labels, len(paragraphs)

def encode_labels(labels):
    labelencoder = LabelEncoder()
    labelencoder.fit(labels)
    classes = labelencoder.classes_
    num_classes = len(classes)
    labels = labelencoder.transform(labels)

    labels = np.asarray([category_to_multiclass(label, num_classes) for label in labels])

    return labels, classes, num_classes

def vectorize_tfidf(paragraphs):
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform(paragraphs)

    return vectors

def main():
    logger.info("Loading base dataset")
    paragraphs, labels, dataset_length = load_train_test()
    logger.success(f"Base dataset length: {dataset_length}")

    logger.info("Encoding labels to multiclass")
    y, classes, num_classes = encode_labels(labels)
    logger.success(f"Classes: {classes}")
    logger.success(f"Shape of y: {y.shape}")
    logger.success(f"Type of y: {type(y)}")
    logger.success(f"Type of y[0]: {type(y[0])}")

    logger.info("Vectorizing base dataset")
    X = vectorize_tfidf(paragraphs)
    logger.success(f"Shape of X: {X.shape}")
    logger.success(f"Type of X; {type(X)}")
    logger.success(f"Type of X[0]: {type(X[0])}")
    logger.info("To array")
    X = X.toarray()
    logger.success(f"Type of X: {type(X)}")
    logger.success(f"Type of X[0]: {type(X[0])}")

    logger.info("Splitting to train test")
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.2)
    # X_train = np.transpose(X_train)
    # X_test = np.transpose(X_test)
    logger.success(f"Shape X_train: {X_train.shape}")
    logger.success(f"Shape of y_train: {y_train.shape}")    
    logger.success(f"Shape of X_test: {X_test.shape}")
    logger.success(f"Shape of y_test: {y_test.shape}")

    logger.info("Initiating Training Loop")
    # training loop
    N = X_train.shape[0]            # Dataset length
    V = X_train.shape[1]            # Vocab length
    M = 5                           # Batch size
    K = num_classes                 # Number of classes
    W = np.random.random((V, K))    # Weight vector
    b = np.random.random((1, K))    # Bias vector
    B = np.repeat(b, M, axis=0)     # Bias matrix
    lr = 1e-2                       # Learning rate
    lmda = 1e-2                     # Regularization coefficient
    logger.success(f"Shape of N (Dataset length): {N}")
    logger.success(f"Shape of V (Vocab length): {V}")
    logger.success(f"Shape of M (Batch size): {M}")
    logger.success(f"Shape of K (Num classes): {K}")
    logger.success(f"Shape of W (Weight vector): {W.shape}")
    logger.success(f"Shape of b (Bias): {b.shape}")
    logger.success(f"Shape of B (Bias matrix for iteration): {B.shape}")


    def get_accuracy(X, Y):
        """
        Calculate the accuracy based off of a set of inputs and their associated labels
        """
        assert(len(X) == len(Y))
        correct = 0
        total = len(X)

        for Xi, Yi in zip(X, Y):
            # if predict(Xi) == np.argmax(Yi):
            if np.argmax(net(Xi, multi=False)) == np.argmax(Yi):
                correct += 1
        
        return correct / total

    def softmax(x, multi=True):
        """
        Compute softmax
        """
        e_x = np.exp(x - np.max(x))

        if multi:
            return e_x / e_x.sum(axis=1)
        else:
            return e_x / e_x.sum(axis=0)

    def regularization(index, k):
        return lmda * (W[index, k]**2)
    
    def regularization_gradient(k):
        return 2 * lmda * W[:, k]

    def net2(Xi, k):
        # Xi (1 x V) W (V x K)
        y_linear = np.add(np.dot(Xi, W[:, k]), b[0, k])
        # y_linear (1 x 1)
        return softmax(y_linear)
    
    def net(Xi, multi=(not (M==1))):
        # Xi (1 x V) W (V x K) b (1 x K)
        y_linear = np.add(np.dot(Xi, W), b[0])
        # y_linear (1 x K)
        return softmax(y_linear, multi=multi)

    epochs = 10
    batches = int(len(X_train) / M)
    dataset_length = y_train.shape[0]
    dataset_indexes = np.arange(dataset_length)

    # yhat has to be made outside of the batch 
    for epoch in range(epochs):
        logger.info(f"Epoch: {epoch}")
        epoch_loss = 0
        epoch_gradient = np.zeros((V, K))
        epoch_accuracy = 0
        # need to step through entire dataset for each epoch based off of the batching
        for n in range(0, N, M):
            # for the entire dataset in batches of M
            # choose M random indexes without replacement from the dataset
            indexes = np.random.choice(dataset_indexes, M, replace=False)
            X = X_train[indexes]
            Y = y_train[indexes]
            # X and Y are our batches of length M
            # Shape of X: (M x V)
            # Shape of Y: (M x K)
            # need to stop through all of the examples in the batch 
            for index, Xi, Yi in zip(indexes, X, Y):
                # Xi is a single example of shape (1 x V)
                # Yi is a single label of shape (1 x K)
                # W is our weight matrix of shape (V x K)
                # yhat is our prediction of shape (1 x K)
                Xi = Xi.reshape(1, V)
                Yi = Yi.reshape(1, K)
                yhat = net(Xi, multi=(not (M==1))).reshape(1, K)
                # logger.info(f"yhat: {yhat}")
                k_true = np.where(Yi[0] == 1)
                # Prediction component of Loss can be easily updated with a matrix operation
                epoch_loss += -(1/N) * (np.dot(np.transpose(Yi), yhat))
                # Prediction component of epoch gradient can be easily updated with a matrix operation
                # only update the column corresponding to the correct output for this step - substitute for product with Yi
                epoch_gradient[:, k_true] += (np.transpose(Xi) @ (1-yhat))[:, k_true]
                # for all classes
                for k in range(K):
                    # regularization part of loss has to be updated per class
                    epoch_loss += regularization(index, k)
                    epoch_gradient[:, k] +=  regularization_gradient(k)
        logger.info(f"weights before: \n{W}")
        W = W - lr * epoch_gradient
        logger.success(f"weights after: \n{W}")
        logger.error(f"Accuracy: {get_accuracy(X_test, y_test)}")
                    

                    

                










    # for epoch in range(epochs):
    #     logger.info(f"Epoch: {epoch}")
    #     epoch_loss = 0
    #     epoch_gradient = np.zeros((M, K))
    #     for i in range(batches):
    #         for j in range(M):
    #             indexes = np.random.choice(dataset_indexes, M, replace=False)
    #             Xi = X_train[indexes]
    #             Yi = y_train[indexes]
    #             yhat = net(Xi)
    #             epoch_loss += -1/N * Yi * yhat + lmda * (W[indexes, :]**2)
    #             # logger.info(f"Epoch loss: {epoch_loss}")
    #             # epoch_gradient += Yi * np.dot(1-net(Xi), Xi) + 2 * lmda * W[M, :]
    #             first_term = np.zeros((M, 1))
    #             # if M > 1:
    #             #     for h in range(M):
    #             #         k = np.where(Yi[h] == 1)
    #             first_term += Yi * np.dot((1-yhat), Xi)
    #             # elif M == 1:
    #             #     first_term = 1 * (1-yhat)[k] * Xi
    #             epoch_gradient += np.add(first_term, Yi * 2 * lmda * W[:, k])
    #             bias_epoch_gradient = (yhat - Yi)
    #             # logger.info(f"Bias epoch gradient: {bias_epoch_gradient}")
    #             # logger.info(f"Epoch gradient: {epoch_gradient}")
    #     W = W - lr * epoch_gradient
    #     B = B - lr * bias_epoch_gradient
    #     logger.success("Weights updated")
    #     logger.info("Testing accuracy")
    #     epoch_accuracy = get_accuracy(X_test, y_test)
    #     logger.success(f"Accuracy at epoch {epoch}: {epoch_accuracy}")

    # # end training loop
    # logger.success("Completed Training Loop!")

    # logger.info(f"Weights shape: {W.shape}")
    # logger.info(f"Bias shape: {B.shape}")
    
    # logger.info("Checking final accuracy")
    # accuracy = get_accuracy(X_test, y_test)
    # logger.success(f"Accuracy: {accuracy}")


    # testX = X_train[0]
    # testY = y_train[0]
    # k = np.where(testY == 1)
    # logger.info(f"testY: {testY}")
    # logger.info(f"k = {k}")
    # net_output = net(testX)
    # logger.info(f"net output: {net_output[0]}")
    # logger.info(f"net prediction: {np.argmax(net_output[0])}")



if __name__ == "__main__":
    main()
