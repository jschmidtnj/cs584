#!/usr/bin/env python3
"""
main file

entry point for running assignment 1
"""

from variables import random_state
import numpy as np
import tensorflow as tf
import random
from loguru import logger
from clean_documents import clean as clean_documents
from clean_reviews import clean as clean_reviews
from classification import train_test as train_test_books
from sentiment import train_test as train_test_reviews


def initialize() -> None:
    """
    initialize before running anything
    """
    tf.random.set_seed(random_state)
    random.seed(random_state)
    np.random.seed(random_state)
    # show if there is a GPU. this will allow for faster training
    logger.info(
        f"Num GPUs Available: {len(tf.config.experimental.list_physical_devices('GPU'))}")


def main() -> None:
    """
    main entry point for program
    """
    initialize()

    # Clean Data
    classification_data, classes_list = clean_documents()
    logger.info(
        f'\nsample of output data:\n{classification_data.sample(n=5)}')

    # Train Models
    # train_test_books(classification_data, classes_list)

    # Clean Data
    sentiment_data = clean_reviews()
    logger.info(
        f'\nsample of output data:\n{sentiment_data.sample(n=5)}')

    # Train and Evaluate
    train_test_reviews(sentiment_data)


if __name__ == '__main__':
    main()
