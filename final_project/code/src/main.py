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


if __name__ == '__main__':
    main()
