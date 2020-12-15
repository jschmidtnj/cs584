#!/usr/bin/env python3
"""
main file

entry point for running final project
"""

from variables import random_state
import numpy as np
import tensorflow as tf
import random
import matplotlib.pyplot as plt
import seaborn as sns
from loguru import logger
from data import read_data
from data_attention import read_data_attention
from embeddings import build_embeddings
from simple_rnn import simple_rnn
from lstm import run_lstm
from gru import run_gru
from rnn import run_rnn
from distilibert import run_distilibert
from roberta import run_roberta

EMBEDDING_SIZE_Y: int = 300
# epochs for attention models
EPOCHS: int = 3


def initialize() -> tf.distribute.TPUStrategy:
    """
    initialize before running anything
    """
    tf.random.set_seed(random_state)
    random.seed(random_state)
    np.random.seed(random_state)
    sns.set_style('whitegrid')
    plt.style.use('fivethirtyeight')

    try:
        # TPU detection
        tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
        logger.info(f'Running on TPU {tpu.master()}')
    except ValueError:
        tpu = None

    if tpu:
        tf.config.experimental_connect_to_cluster(tpu)
        tf.tpu.experimental.initialize_tpu_system(tpu)
        strategy = tf.distribute.experimental.TPUStrategy(tpu)
    else:
        strategy = tf.distribute.get_strategy()

    logger.info(f'Num Replicas: {strategy.num_replicas_in_sync}')

    # show if there is a GPU. this will allow for faster training
    logger.info(
        f"Num GPUs Available: {len(tf.config.experimental.list_physical_devices('GPU'))}")

    return strategy


def main() -> None:
    """
    main entry point for program
    """
    strategy = initialize()

    # read in the data
    x_train_padded, x_valid_padded, y_train, y_valid, max_len, word_indexes = read_data()

    # run base models
    simple_rnn(strategy, x_train_padded, x_valid_padded, y_train,
               y_valid, max_len, len(word_indexes) + 1, EMBEDDING_SIZE_Y)
    embeddings_output = build_embeddings(EMBEDDING_SIZE_Y, word_indexes)
    run_lstm(strategy, x_train_padded, x_valid_padded, y_train, y_valid,
             max_len, len(word_indexes) + 1, EMBEDDING_SIZE_Y, embeddings_output)
    run_gru(strategy, x_train_padded, x_valid_padded, y_train, y_valid,
            max_len, len(word_indexes) + 1, EMBEDDING_SIZE_Y, embeddings_output)
    run_rnn(strategy, x_train_padded, x_valid_padded, y_train, y_valid,
            max_len, len(word_indexes) + 1, EMBEDDING_SIZE_Y, embeddings_output)

    attention_max_len = 192

    # read in attention data
    x_train, x_valid, y_train, y_valid, train_dataset, \
        valid_dataset, test_dataset, batch_size = read_data_attention(
            strategy, attention_max_len)
    run_distilibert(strategy, x_train, x_valid, y_train, y_valid,
                    train_dataset, valid_dataset, test_dataset, attention_max_len, EPOCHS, batch_size)

    x_train, x_valid, y_train, y_valid, train_dataset, \
        valid_dataset, test_dataset, batch_size = read_data_attention(
            strategy, attention_max_len)
    run_roberta(strategy, x_train, x_valid, y_train, y_valid,
                train_dataset, valid_dataset, test_dataset, attention_max_len, EPOCHS, batch_size)


if __name__ == '__main__':
    main()
