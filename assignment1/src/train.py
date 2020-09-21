#!/usr/bin/env python3
"""
training
"""

from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neural_network import MLPClassifier
from variables import paragraph_key, class_key, random_state
from books import class_map
import pandas as pd
import numpy as np
from loguru import logger
from typing import List
import matplotlib.pyplot as plt

TEST_SIZE = 0.2
NUM_EPOCHS = 1 * 3
PLOT_EPOCH_ITER = 1
BATCH_SIZE = 128

assert NUM_EPOCHS >= PLOT_EPOCH_ITER, 'number of epochs must be greater than plot iter'

MIN_EPOCHS = 1000
# assert NUM_EPOCHS > MIN_EPOCHS, f'num epochs must be greater than {MIN_EPOCHS} to get convergence'

# https://www.kaggle.com/sudhirnl7/logistic-regression-tfidf
# https://www.kaggle.com/kashnitsky/logistic-regression-tf-idf-baseline

# plot learning curves: https://scikit-learn.org/stable/auto_examples/model_selection/plot_learning_curve.html

def one_minus(data: List[float]) -> List[float]:
    """
    return 1 - each element in list
    """
    return list(map(lambda elem: 1 - elem, data))

def train(clean_data: pd.DataFrame) -> None:
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
        solver='lbfgs', penalty='l2', random_state=random_state, n_jobs=num_splits, verbose=False)

    # by default, the mlp loss function is log_loss
    optimizer: str = 'adam'
    learning_rate: float = 0.001
    mlp = MLPClassifier(random_state=random_state,
                        hidden_layer_sizes=(3,), verbose=False,
                        learning_rate='constant', learning_rate_init=learning_rate,
                        solver=optimizer, max_iter=1000)

    X = clean_data[paragraph_key].values
    y = clean_data[class_key].values

    mlp_training_scores: List[int] = []
    mlp_testing_scores: List[int] = []

    for train_index, test_index in skf.split(X, y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        logger.info('test train split complete')

        logger.info('start text transform')
        text_transformer = TfidfVectorizer(stop_words='english', ngram_range=(
            1, 2), lowercase=True, max_features=150000)
        X_train_text = text_transformer.fit_transform(X_train)
        X_test_text = text_transformer.transform(X_test)
        logger.info('text transformed')

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
                    X_train_text[current_indices], y_train[current_indices], classes=classes)
            if (epoch + 1) % PLOT_EPOCH_ITER == 0:
                mlp_training_scores.append(mlp.score(X_train_text, y_train))
                mlp_testing_scores.append(mlp.score(X_test_text, y_test))
        logger.info(f'mlp testing score: {mlp_testing_scores[-1]}')

    # TODO - get cross validation score. is this part of skf?

    fig, ax = plt.subplots(2, sharex=True, sharey=True)
    ax[0].plot(one_minus(mlp_training_scores))
    ax[0].set_title('Training Loss')
    ax[1].plot(one_minus(mlp_testing_scores))
    ax[1].set_title('Validation Loss')
    fig.suptitle(
        f"Training and Validation Loss every {PLOT_EPOCH_ITER} epoch{'' if PLOT_EPOCH_ITER == 1 else 's'}", fontsize=14)
    plt.show()


if __name__ == '__main__':
    raise RuntimeError("training cannot be run on its own")
