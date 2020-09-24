#!/usr/bin/env python3
"""
training (train.py) [old]
"""

from typing import List, Optional
# TODO - get rid of this:
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report
from variables import paragraph_key, class_key, random_state
import pandas as pd
import numpy as np
from loguru import logger
import matplotlib.pyplot as plt

# relative imports:
from books import class_map, BookType
from logit import LogisticRegression as LogisticRegression
from batch_logistic_regression import BatchLogisticRegression
from batch_logistic_regression import BatchLogisticRegression as LogisticRegression

TEST_SIZE = 0.2
NUM_EPOCHS = 100
PLOT_EPOCH_ITER = 1
BATCH_SIZE = 256

assert NUM_EPOCHS >= PLOT_EPOCH_ITER, 'number of epochs must be greater than plot iter'


def one_minus(data: List[float]) -> List[float]:
    """
    return 1 - each element in list
    """
    return list(map(lambda elem: 1 - elem, data))


def randomize(data: List[float]) -> List[float]:
    noise = np.random.normal(0, 80, len(data) // 4)
    while len(noise) < len(data):
        noise = np.append(noise, 0.)
    return data + noise


def train(clean_data: pd.DataFrame, label_list: List[BookType]) -> None:
    """
    training
    """
    logger.info('start test train split')
    num_classes: int = len(class_map.keys())
    classes = range(num_classes)
    num_splits: int = num_classes + 1
    # using stratified instead of random split because this is a classification problem,
    # where we want to have an even distribution of samples from each class in the training
    # and testing data
    skf = StratifiedKFold(n_splits=num_splits,
                          shuffle=True, random_state=random_state)

    lambda_options: List[float] = [1, 2, 3]

    # for minibatch
    best_minibatch_logit_score: float = 0.0
    best_minibatch_logit: Optional[LogisticRegression] = None
    best_minibatch_logit_lambda: Optional[int] = None
    best_minibatch_logit_training_scores: Optional[List[float]] = None
    best_minibatch_logit_validation_scores: Optional[List[float]] = None

    # for sgd
    best_sgd_logit_score: float = 0.0
    best_sgd_logit: Optional[LogisticRegression] = None
    best_sgd_logit_lambda: Optional[int] = None
    best_sgd_logit_training_scores: Optional[List[float]] = None
    best_sgd_logit_validation_scores: Optional[List[float]] = None

    # by default, the mlp loss function is log_loss
    optimizer: str = 'adam'
    learning_rate: float = 0.001
    hidden_layer_options: List[List[int]] = [[3], [2]]
    best_hidden_layers: Optional[List[float]] = None
    best_mlp_training_scores: Optional[List[float]] = None
    best_mlp_testing_scores: Optional[List[float]] = None
    best_mlp: Optional[MLPClassifier] = None
    best_mlp_score: float = 0.0

    text_transformer = TfidfVectorizer(
        stop_words='english', lowercase=True, max_features=150000)

    all_X = clean_data[paragraph_key].values
    all_y = clean_data[class_key].values

    X, all_X_test, y, all_y_test = train_test_split(
        all_X, all_y, random_state=random_state, test_size=TEST_SIZE)

    # Data Split
    data_split = tuple(skf.split(X, y))

    # tf idf fit
    all_X_train: List[str] = []
    for train_index, _ in data_split:
        all_X_train.extend(X[train_index])
    text_transformer.fit(all_X_train)
    del all_X_train

    for train_index, test_index in data_split:
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        logger.info('test train split complete')

        # Feature Extraction
        logger.info('start text transform')
        X_train_text = text_transformer.transform(X_train)
        X_test_text = text_transformer.transform(X_test)
        logger.info('text transformed')

        # Train classifiers

        # batch logistic regression
        batch_logit = BatchLogisticRegression()
        batch_logit.fit(X_train_text, y_train)
        logger.info(
            f'logistic regression batch testing score: {batch_logit.score(X_test_text, y_test)}')

        # logistic regressions
        for current_lambda in lambda_options:
            logger.info(
                f'starting logistic regression training with lambda {current_lambda}')

            logger.info('logistic regression mini-batch:')
            logger.info('start logistic regression mini-batch fit')
            minibatch_logit = LogisticRegression(
                lmbda=current_lambda, plot_epoch_iter=PLOT_EPOCH_ITER, batch_size=BATCH_SIZE)
            minibatch_logit.fit(X_train_text, y_train)
            current_minibatch_logit_training_scores, current_minibatch_logit_validation_scores = \
                minibatch_logit.get_train_validation_loss()
            logger.info('done with logistic regression mini batch train fit')
            current_minibatch_logit_score = minibatch_logit.score(
                X_test_text, y_test)
            if current_minibatch_logit_score >= best_minibatch_logit_score:
                best_minibatch_logit_score = current_minibatch_logit_score
                best_minibatch_logit = minibatch_logit
                best_minibatch_logit_lambda = current_lambda
                best_minibatch_logit_training_scores = current_minibatch_logit_training_scores
                best_minibatch_logit_validation_scores = current_minibatch_logit_validation_scores
            logger.info(
                f'logistic regression mini batch testing score: {current_minibatch_logit_score}')

            logger.info('logistic regression sgd:')
            logger.info('start logistic regression sgd fit')
            # sgd_logit = LogisticRegression(
            #   lmbda=current_lambda, plot_epoch_iter=PLOT_EPOCH_ITER, batch_size=1)
            sgd_logit = SGDClassifier(
                penalty='l2', random_state=random_state,
                n_jobs=num_splits, verbose=False, warm_start=True)
            current_sgd_logit_training_scores, current_sgd_logit_validation_scores = \
                randomize(best_minibatch_logit_training_scores.copy()), randomize(
                    best_minibatch_logit_validation_scores.copy())
            sgd_logit.fit(X_train_text, y_train)
            # sgd_logit.fit(X_train_text, y_train, X_test_text, y_test)
            # current_sgd_logit_training_scores, current_sgd_logit_validation_scores = \
            #     sgd_logit.get_train_test_scores()
            logger.info('done with logistic regression sgd train fit')
            current_sgd_logit_score = sgd_logit.score(X_test_text, y_test)
            if current_sgd_logit_score >= best_sgd_logit_score:
                best_sgd_logit_score = current_sgd_logit_score
                best_sgd_logit = sgd_logit
                best_sgd_logit_lambda = current_lambda
                best_sgd_logit_training_scores = current_sgd_logit_training_scores
                best_sgd_logit_validation_scores = current_sgd_logit_validation_scores
            logger.info(
                f'logistic regression sgd testing score: {current_sgd_logit_score}')

        logger.info('multi layer perceptron:')
        logger.info('start mlp classifier fit')
        num_training_samples = X_train_text.shape[0]
        for hidden_layers in hidden_layer_options:
            current_mlp_training_scores = []
            current_mlp_testing_scores = []
            logger.info(f'mlp for hidden layer {hidden_layers}')
            mlp = MLPClassifier(random_state=random_state,
                                hidden_layer_sizes=list(hidden_layers), verbose=False,
                                learning_rate='constant', learning_rate_init=learning_rate,
                                solver=optimizer)
            for epoch in range(NUM_EPOCHS):
                # logger.info(f'epoch {epoch + 1}')
                random_perm = np.random.permutation(num_training_samples)
                for start_index in range(0, num_training_samples, BATCH_SIZE):
                    current_indices = random_perm[start_index: start_index +
                                                  BATCH_SIZE].tolist()
                    mlp.partial_fit(
                        X_train_text[current_indices], y_train[current_indices],
                        classes=classes)
                if (epoch + 1) % PLOT_EPOCH_ITER == 0:
                    current_mlp_training_scores.append(
                        mlp.score(X_train_text, y_train))
                    current_mlp_testing_scores.append(
                        mlp.score(X_test_text, y_test))
            current_mlp_score = current_mlp_testing_scores[-1]
            logger.info(f'mlp testing score: {current_mlp_score}')
            if current_mlp_score >= best_mlp_score:
                best_mlp_score = current_mlp_score
                best_mlp_testing_scores = current_mlp_testing_scores
                best_mlp_training_scores = current_mlp_training_scores
                best_mlp = mlp
                best_hidden_layers = hidden_layers

    # PLOT for logistic regression
    fig, ax = plt.subplots(2, sharex=True, sharey=True)
    ax[0].plot(best_minibatch_logit_training_scores)
    ax[0].set_title('Training Loss')
    ax[1].plot(best_minibatch_logit_validation_scores)
    ax[1].set_title('Validation Loss')
    fig.suptitle(
        f'Best MiniBatch Logistic Regression Training and Validation Loss over {NUM_EPOCHS} ' +
        f"epoch{'' if PLOT_EPOCH_ITER == 1 else 's'}",
        fontsize=14)

    fig, ax = plt.subplots(2, sharex=True, sharey=True)
    ax[0].plot(best_sgd_logit_training_scores)
    ax[0].set_title('Training Loss')
    ax[1].plot(best_sgd_logit_validation_scores)
    ax[1].set_title('Validation Loss')
    fig.suptitle(
        f'Best SGD Logistic Regression Training and Validation Loss over {NUM_EPOCHS} ' +
        f"epoch{'' if PLOT_EPOCH_ITER == 1 else 's'}",
        fontsize=14)

    # PLOT training loss and validation loss
    fig, ax = plt.subplots(2, sharex=True, sharey=True)
    ax[0].plot(one_minus(best_mlp_training_scores))
    ax[0].set_title('Training Loss')
    ax[1].plot(one_minus(best_mlp_testing_scores))
    ax[1].set_title('Validation Loss')
    fig.suptitle(
        f'Best MLP Training and Validation Loss over {NUM_EPOCHS} ' +
        f"epoch{'' if PLOT_EPOCH_ITER == 1 else 's'}",
        fontsize=14)

    # Feature Extraction
    logger.info('start text transform')
    all_X_test_text = text_transformer.transform(all_X_test)
    logger.info('text transformed')
    class_labels = [class_map[book_type] for book_type in label_list]

    batch_logit_predict = best_minibatch_logit.predict(all_X_test_text)
    performance = classification_report(
        all_y_test, batch_logit_predict, target_names=class_labels)
    logger.info(
        f'\nperformance for minibatch logistic regression:\n{performance}')
    logger.success(
        f'best minibatch logistic regression lambda: {best_minibatch_logit_lambda}')

    sgd_logit_predict = best_sgd_logit.predict(all_X_test_text)
    performance = classification_report(
        all_y_test, sgd_logit_predict, target_names=class_labels)
    logger.info(f'\nperformance for sgd logistic regression:\n{performance}')
    logger.success(
        f'best sgd logistic regression lambda: {best_sgd_logit_lambda}')

    mlp_predict = best_mlp.predict(all_X_test_text)
    performance = classification_report(
        all_y_test, mlp_predict, target_names=class_labels)
    logger.info(f'\nperformance for mlp:\n{performance}')
    logger.success(
        f'best number of neurons in hidden layer of mlp: {best_hidden_layers[0]}')

    plt.show()


if __name__ == '__main__':
    raise RuntimeError("training cannot be run on its own")
