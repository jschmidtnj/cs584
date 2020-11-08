#!/usr/bin/env python3
"""
classification (classification.py)
"""

from __future__ import annotations

import string
import yaml
import re
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
from books import BookType

from tensorflow.keras.layers.experimental.preprocessing import TextVectorization


def standardize_text(text: tf.Tensor):
    lowercase = tf.strings.lower(text)
    without_spaces = tf.strings.strip(lowercase)
    return tf.strings.regex_replace(without_spaces,
                                    '[%s]' % re.escape(string.punctuation),
                                    '')

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

    vocab_size = 10000
    sequence_length = 250

    vectorize_layer = TextVectorization(
        standardize=standardize_text,
        max_tokens=vocab_size,
        output_mode='int',
        output_sequence_length=sequence_length)

    autotune = tf.data.experimental.AUTOTUNE
    training_dataset = training_dataset.cache().prefetch(buffer_size=autotune)

    train_text = training_dataset.map(lambda x, y: x)
    vectorize_layer.adapt(train_text)

    embedding_dim = 16

    embedding_layer = tf.keras.layers.Embedding(vocab_size, embedding_dim)
    output_layer = tf.keras.layers.Dense(len(BookType.get_values()))

    lstm_model = tf.keras.models.Sequential([
        vectorize_layer,
        embedding_layer,
        tf.keras.layers.LSTM(128),
        tf.keras.layers.Dense(64, activation='relu'),
        output_layer,
    ])

    learning_rate: int = 1e-3
    optimizer = tf.keras.optimizers.Adam(learning_rate)
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    lstm_model.compile(optimizer=optimizer,
                        loss=loss,
                        metrics=['accuracy'])

    # lstm_model.fit(
    #     training_dataset,
    #     epochs=15,
    #     callbacks=[])

    # logger.info('lstm model summary:')
    # lstm_model.summary()

    # running lstm_model.predict() will give the last hidden state
    # TODO - need to figure out how to get the max hidden state

    cnn_model = tf.keras.models.Sequential([
        vectorize_layer,
        embedding_layer,
        tf.keras.layers.ZeroPadding1D(1),
        tf.keras.layers.Conv1D(32, 3, activation='relu'),
        tf.keras.layers.MaxPooling1D(3),
        tf.keras.layers.Conv1D(32, 2, activation='relu'),
        tf.keras.layers.MaxPooling1D(2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(64, activation='relu'),
        output_layer,
    ])

    cnn_model.compile(optimizer=optimizer,
                    loss=loss,
                    metrics=['accuracy'])

    cnn_model.fit(training_dataset, epochs=10, 
                callbacks=[])

    cnn_model.summary()


if __name__ == '__main__':
    if len(argv) < 2:
        raise ValueError('no n-grams training data provided')
