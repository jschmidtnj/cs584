#!/usr/bin/env python3
"""
training
"""

from typing import List
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report
from variables import paragraph_key, class_key, random_state
from books import class_map, BookType
import pandas as pd
import numpy as np
from loguru import logger
import matplotlib.pyplot as plt

TEST_SIZE = 0.2
NUM_EPOCHS = 10 * 3
PLOT_EPOCH_ITER = 1
BATCH_SIZE = 256

assert NUM_EPOCHS >= PLOT_EPOCH_ITER, 'number of epochs must be greater than plot iter'

MIN_EPOCHS = 1000
# assert NUM_EPOCHS > MIN_EPOCHS, f'num epochs must be greater than {MIN_EPOCHS} to get convergence'

# https://www.kaggle.com/sudhirnl7/logistic-regression-tfidf
# https://www.kaggle.com/kashnitsky/logistic-regression-tf-idf-baseline


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

    logit = LogisticRegression(
        solver='lbfgs', penalty='l2', random_state=random_state,
        n_jobs=num_splits, verbose=False, warm_start=True)

    # by default, the mlp loss function is log_loss
    optimizer: str = 'adam'
    learning_rate: float = 0.001
    hidden_layer_options: List[List[int]] = [[3]]
    best_hidden_layers: List[int] = hidden_layer_options[0]
    mlp = MLPClassifier(random_state=random_state,
                        hidden_layer_sizes=list(best_hidden_layers), verbose=False,
                        learning_rate='constant', learning_rate_init=learning_rate,
                        solver=optimizer)

    text_transformer = TfidfVectorizer(stop_words='english', ngram_range=(
        1, 2), lowercase=True, max_features=150000)

    X = clean_data[paragraph_key].values
    y = clean_data[class_key].values

    mlp_training_scores: List[float] = []
    mlp_testing_scores: List[float] = []

    all_X_test: List[str] = []
    all_y_test: List[float] = []

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
        all_X_test.extend(X_test.tolist())
        y_train, y_test = y[train_index], y[test_index]
        all_y_test.extend(y_test)
        logger.info('test train split complete')

        # Feature Extraction
        logger.info('start text transform')
        X_train_text = text_transformer.transform(X_train)
        X_test_text = text_transformer.transform(X_test)
        logger.info('text transformed')

        # Train classifiers
        logger.info('logistic regression:')
        logger.info('start logistic regression fit')
        logit.fit(X_train_text, y_train)
        logger.info('done with logistic regression train fit')
        logger.info(
            f'logistic regression testing score: {logit.score(X_test_text, y_test)}')

        logger.info('multi layer perceptron:')
        logger.info('start mlp classifier fit')
        num_training_samples = X_train_text.shape[0]
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
                mlp_training_scores.append(mlp.score(X_train_text, y_train))
                mlp_testing_scores.append(mlp.score(X_test_text, y_test))
        logger.info(f'mlp testing score: {mlp_testing_scores[-1]}')

    # PLOT training loss and validation loss
    fig, ax = plt.subplots(2, sharex=True, sharey=True)
    ax[0].plot(one_minus(mlp_training_scores))
    ax[0].set_title('Training Loss')
    ax[1].plot(one_minus(mlp_testing_scores))
    ax[1].set_title('Validation Loss')
    fig.suptitle(
        f"Training and Validation Loss every {PLOT_EPOCH_ITER} epoch{'' if PLOT_EPOCH_ITER == 1 else 's'}", fontsize=14)

    # Feature Extraction
    logger.info('start text transform')
    all_X_test_text = text_transformer.transform(all_X_test)
    logger.info('text transformed')
    class_labels = [class_map[book_type] for book_type in label_list]
    logit_predict = logit.predict(all_X_test_text)
    performance = classification_report(
        all_y_test, logit_predict, target_names=class_labels)
    logger.info(f'\nperformance for logistic regression:\n{performance}')
    mlp_predict = mlp.predict(all_X_test_text)
    # TODO - get lambda for logistic regression
    logger.info(f'logistic regression lambda: {"TODO"}')
    performance = classification_report(
        all_y_test, mlp_predict, target_names=class_labels)
    logger.info(f'\nperformance for mlp:\n{performance}')
    logger.info(
        f'number of neurons in hidden layer of mlp: {best_hidden_layers[0]}')

    plt.show()


if __name__ == '__main__':
    raise RuntimeError("training cannot be run on its own")
