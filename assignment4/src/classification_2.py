#!/usr/bin/env python3
"""
classification (classification.py)
"""

from __future__ import annotations

import string
import yaml
from os.path import exists
from sys import argv
from typing import List, Optional, Any, Dict
from utils import file_path_relative
from variables import paragraph_key, clean_data_folder, models_folder, \
    cnn_folder, text_vectorization_folder, cnn_file_name, class_key
from loguru import logger
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa

from tensorflow.keras.layers.experimental.preprocessing import TextVectorization

# read dataset in batches of
batch_size = 10

def cnn_train(name: str, clean_data: pd.DataFrame) -> tf.keras.models.Sequential:
    """
    cnn training
    creates the tensorflow cnn model for word prediction
    """
    logger.info(f'run cnn training for {name}')

    all_paragraphs: List[str] = [' '.join(paragraph) for paragraph in clean_data[paragraph_key]]
    labels: List[int] = clean_data[class_key]
    training_dataset = tf.data.Dataset.from_tensor_slices((all_paragraphs, labels))

    # buffer size is used to shuffle the dataset
    buffer_size = 10000
    training_dataset = training_dataset.shuffle(buffer_size).batch(
        batch_size, drop_remainder=True)

    # print some samples
    logger.success('training data sample:')
    for input_example, target_example in training_dataset.take(1):
        logger.info(f"\ninput: {input_example}\ntarget: {target_example}")

    autotune = tf.data.experimental.AUTOTUNE
    training_dataset = training_dataset.cache().prefetch(buffer_size=autotune)


if __name__ == '__main__':
    if len(argv) < 2:
        raise ValueError('no n-grams training data provided')
