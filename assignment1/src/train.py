#!/usr/bin/env python3
"""
training
"""

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neural_network import MLPClassifier
from variables import paragraph_key, class_key, random_state
import pandas as pd
from loguru import logger

TEST_SIZE = 0.2

# https://www.kaggle.com/sudhirnl7/logistic-regression-tfidf
# https://www.kaggle.com/kashnitsky/logistic-regression-tf-idf-baseline

# plot learning curves: https://scikit-learn.org/stable/auto_examples/model_selection/plot_learning_curve.html


def train(clean_data: pd.DataFrame) -> None:
    """
    training
    """
    logger.info('start test train split')
    X_train, X_test, y_train, y_test = train_test_split(
        clean_data[paragraph_key], clean_data[class_key], test_size=TEST_SIZE, random_state=random_state)
    logger.info('test train split complete')
    logger.info('start text transform')
    text_transformer = TfidfVectorizer(stop_words='english', ngram_range=(
        1, 2), lowercase=True, max_features=150000)
    X_train_text = text_transformer.fit_transform(X_train)
    X_test_text = text_transformer.transform(X_test)
    logger.info('text transformed')

    logger.info('logistic regression:')
    logger.info('start cross validation')
    logit = LogisticRegression(
        solver='lbfgs', penalty='l2', random_state=random_state, n_jobs=4, verbose=False)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    # TODO - figure out how to do this based on the classes
    # I don't think I should be using this. the directions say
    # to do it on the training data separately
    cv_results = cross_val_score(
        logit, X_train_text, y_train, cv=skf, scoring='f1_micro')
    logger.info(f'cross-validation results: {cv_results}')
    logger.info('start logistic regression fit')
    logit.fit(X_train_text, y_train)
    logger.info('done with logistic regression train fit')
    logger.info(f'testing score: {logit.score(X_test_text, y_test)}')

    logger.info('multi layer perceptron:')
    logger.info('start mlp classifier fit')
    clf = MLPClassifier(random_state=random_state,
                        hidden_layer_sizes=(3,), max_iter=100, verbose=False)
    clf.fit(X_train_text, y_train)
    logger.info(f'testing score: {clf.score(X_test_text, y_test)}')


if __name__ == '__main__':
    raise RuntimeError("training cannot be run on its own")
