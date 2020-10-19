#!/usr/bin/env python3
"""
main file

entry point for running assignment 1
"""

from clean import clean_tokenize
from ngrams import n_grams_train, n_grams_predict_next, SmoothingType
from variables import random_state
import numpy as np
import tensorflow as tf
import random
from rnn import rnn_train, rnn_predict_next
from loguru import logger


def initialize() -> None:
    """
    initialize before running anything
    """
    tf.random.set_seed(random_state)
    random.seed(random_state)
    np.random.seed(random_state)
    logger.info(
        f"Num GPUs Available: {len(tf.config.experimental.list_physical_devices('GPU'))}")


def main() -> None:
    """
    main entry point
    """
    initialize()

    # Clean Data

    train_name: str = 'train'
    train_data = clean_tokenize(f'{train_name}.txt')[0]
    valid_name: str = 'valid'
    validation_data = clean_tokenize(f'{valid_name}.txt')[0]
    input_name: str = 'input'
    input_data = clean_tokenize(f'{input_name}.txt')[0]
    num_predict_input = 30

    # N-Grams

    # train
    # n_grams_train(train_name, clean_data=train_data)
    # test
    # n_grams_predict_next(valid_name, clean_input_data=validation_data,
    #                      smoothing=SmoothingType.basic)
    # n_grams_predict_next(valid_name, clean_input_data=validation_data,
    #                      smoothing=SmoothingType.good_turing)
    # n_grams_predict_next(valid_name, clean_input_data=validation_data,
    #                      smoothing=SmoothingType.kneser_ney)
    # check kneser-ney with input because it has unseen data
    # n_grams_predict_next(input_name, clean_input_data=input_data,
    #                      smoothing=SmoothingType.kneser_ney,
    #                      num_lines_predict=num_predict_input)

    # RNN:
    # train
    # rnn_text_vectorization_model = rnn_train(train_name, clean_data=train_data)
    # test
    # rnn_predict_next(train_name, clean_input_data=validation_data)
    # rnn_predict_next(train_name, clean_input_data=input_data,
    #                  num_lines_predict=num_predict_input)


if __name__ == '__main__':
    main()
