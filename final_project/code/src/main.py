#!/usr/bin/env python3
"""
main file

entry point for running final project
"""

from variables import random_state
import numpy as np
import tensorflow as tf
import random
from loguru import logger
from data import read_data

import matplotlib.pyplot as plt
import seaborn as sns


def initialize():
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
    read_data(strategy)


if __name__ == '__main__':
    main()
