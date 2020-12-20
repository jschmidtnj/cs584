#!/usr/bin/env python3
"""
main file

entry point for running final project
"""

import numpy as np
import tensorflow as tf
import random
import matplotlib.pyplot as plt
import seaborn as sns
from loguru import logger
from utils import file_path_relative
from variables import model_folder, random_state
from data import read_data
from encoder import Encoder
from decoder import Decoder
from train import run_train
from test_model import run_tests


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

    dataset_size: int = 30000
    input_tensor_train, target_tensor_train, input_language, target_language, max_length_target, max_length_input, input_vals, target_vals = read_data(
        dataset_size)

    BUFFER_SIZE = len(input_tensor_train)
    BATCH_SIZE = 64 * strategy.num_replicas_in_sync
    EPOCHS = 15
    steps_per_epoch = len(input_tensor_train) // BATCH_SIZE
    embedding_dim = 256
    units = 1024
    vocab_input_size = len(input_language.word_index) + 1
    vocab_target_size = len(target_language.word_index) + 1

    model_name: str = 'model_1'

    checkpoint_dir = file_path_relative(f'{model_folder}/{model_name}')

    with strategy.scope():
        optimizer = tf.keras.optimizers.Adam()
        encoder = Encoder(vocab_input_size, embedding_dim, units, BATCH_SIZE)
        decoder = Decoder(vocab_target_size, embedding_dim, units, BATCH_SIZE)

        checkpoint = tf.train.Checkpoint(optimizer=optimizer,
                                         encoder=encoder,
                                         decoder=decoder)

        run_train(input_tensor_train, target_tensor_train, target_language, checkpoint, checkpoint_dir,
                  encoder, optimizer, decoder, steps_per_epoch, BUFFER_SIZE, BATCH_SIZE, EPOCHS, model_name)

    # restoring the latest checkpoint in checkpoint_dir
    checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))

    # run tests and get score
    run_tests(max_length_target, max_length_input, input_language, target_language,
              units, encoder, decoder, input_vals, target_vals, model_name)

    # second model
    embedding_dim = 512
    units = 2048

    model_name: str = 'model_2'

    checkpoint_dir = file_path_relative(f'{model_folder}/{model_name}')

    with strategy.scope():
        optimizer_2 = tf.keras.optimizers.Adam()
        encoder_2 = Encoder(vocab_input_size, embedding_dim,
                            units, BATCH_SIZE, gru=True)
        decoder_2 = Decoder(vocab_target_size, embedding_dim,
                            units, BATCH_SIZE, gru=True)

        checkpoint_2 = tf.train.Checkpoint(optimizer=optimizer_2,
                                           encoder=encoder_2,
                                           decoder=decoder_2)

        run_train(input_tensor_train, target_tensor_train, target_language, checkpoint_2, checkpoint_dir,
                  encoder_2, optimizer_2, decoder_2, steps_per_epoch, BUFFER_SIZE, BATCH_SIZE, EPOCHS, model_name)

    # restoring the latest checkpoint in checkpoint_dir
    checkpoint_2.restore(tf.train.latest_checkpoint(checkpoint_dir))

    # run tests and get score
    run_tests(max_length_target, max_length_input, input_language, target_language,
              units, encoder_2, decoder_2, input_vals, target_vals, model_name)


if __name__ == '__main__':
    main()
