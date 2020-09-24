#!/usr/bin/env python3
"""
training
"""

from typing import List, Optional
from sklearn.model_selection import StratifiedKFold
# from sklearn.linear_model import LogisticRegression as LogisticRegression
from sklearn.linear_model import SGDClassifier as SGDClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report
from variables import paragraph_key, class_key, random_state
from books import class_map, BookType
import pandas as pd
import numpy as np
from loguru import logger
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
# from logit import LogisticRegression as LogisticRegression
# from logit2 import LogisticRegression as LogisticRegression2
from logit3 import LogisticRegression as SGDClassifier
# from logit4 import LogisticRegression as LogisticRegression4
from batch_logistic_regression import LogisticRegression as LogisticRegression

TEST_SIZE = 0.2
NUM_EPOCHS = 5
PLOT_EPOCH_ITER = 1
BATCH_SIZE = 256

assert NUM_EPOCHS >= PLOT_EPOCH_ITER, 'number of epochs must be greater than plot iter'


def one_minus(data: List[float]) -> List[float]:
    """
    return 1 - each element in list
    """
    return list(map(lambda elem: 1 - elem, data))


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
    best_logit_score: float = 0.0
    best_logit: Optional[LogisticRegression] = None
    best_logit_lambda: Optional[int] = None
    best_logit_training_losses: Optional[List[float]] = None
    best_logit_validation_losses: Optional[List[float]] = None

    sgd_logit = SGDClassifier(
        penalty='l2', random_state=random_state,
        n_jobs=num_splits, verbose=False, warm_start=True
    )

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
        logger.info('logistic regression batch:')
        logger.info('start logistic regression batch fit')
        logit = LogisticRegression(Cs=lambda_options)
        # logit = LogisticRegression(
        #     solver='lbfgs', penalty='l2', random_state=random_state,
        #     n_jobs=num_splits, verbose=False, warm_start=True)
        current_logit_training_losses, current_logit_validation_losses, current_logit_batch_lambda = logit.fit(X_train_text, y_train)
        logger.info('done with logistic regression batch train fit')
        current_logit_score = logit.score(X_test_text, y_test)
        if current_logit_score >= best_logit_score:
            best_logit_score = current_logit_score
            best_logit = logit
            best_logit_lambda = current_logit_batch_lambda
            best_logit_training_losses = current_logit_training_losses
            best_logit_validation_losses = current_logit_validation_losses
        logger.info(
            f'logistic regression batch testing score: {current_logit_score}')

        logger.info('start logistic regression sgd fit')
        sgd_logit.fit(X_train_text, y_train)
        logger.info('done with logistic regression sgd train fit')
        logger.info(
            f'logistic regression sgd testing score: {sgd_logit.score(X_test_text, y_test)}')

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
                logger.info(f'epoch {epoch + 1}')
                random_perm = np.random.permutation(num_training_samples)
                for start_index in range(0, num_training_samples, BATCH_SIZE):
                    current_indices = random_perm[start_index: start_index +
                                                BATCH_SIZE].tolist()
                    mlp.partial_fit(
                        X_train_text[current_indices], y_train[current_indices],
                        classes=classes)
                if (epoch + 1) % PLOT_EPOCH_ITER == 0:
                    current_mlp_training_scores.append(mlp.score(X_train_text, y_train))
                    current_mlp_testing_scores.append(mlp.score(X_test_text, y_test))
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
    ax[0].plot(best_logit_training_losses)
    ax[0].set_title('Training Loss')
    ax[1].plot(best_logit_validation_losses)
    ax[1].set_title('Validation Loss')
    fig.suptitle(
        f'Best Batch Logistic Regression Training and Validation Loss every {PLOT_EPOCH_ITER} ' +
        f"epoch{'' if PLOT_EPOCH_ITER == 1 else 's'}",
        fontsize=14)

    # PLOT training loss and validation loss
    fig, ax = plt.subplots(2, sharex=True, sharey=True)
    ax[0].plot(one_minus(best_mlp_training_scores))
    ax[0].set_title('Training Loss')
    ax[1].plot(one_minus(best_mlp_testing_scores))
    ax[1].set_title('Validation Loss')
    fig.suptitle(
        f'Best MLP Training and Validation Loss every {PLOT_EPOCH_ITER} ' +
        f"epoch{'' if PLOT_EPOCH_ITER == 1 else 's'}",
        fontsize=14)

    # Feature Extraction
    logger.info('start text transform')
    all_X_test_text = text_transformer.transform(all_X_test)
    logger.info('text transformed')
    class_labels = [class_map[book_type] for book_type in label_list]

    batch_logit_predict = best_logit.predict(all_X_test_text)
    performance = classification_report(
        all_y_test, batch_logit_predict, target_names=class_labels)
    logger.info(f'\nperformance for batch logistic regression:\n{performance}')
    logger.success(f'best batch logistic regression lambda: {best_logit_lambda}')

    mlp_predict = best_mlp.predict(all_X_test_text)
    performance = classification_report(
        all_y_test, mlp_predict, target_names=class_labels)
    logger.info(f'\nperformance for mlp:\n{performance}')
    logger.success(
        f'best number of neurons in hidden layer of mlp: {best_hidden_layers[0]}')

    plt.show()


if __name__ == '__main__':
    raise RuntimeError("training cannot be run on its own")
