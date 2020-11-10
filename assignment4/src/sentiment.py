#!/usr/bin/env python3
"""
classification (classification.py)
"""

from __future__ import annotations

from sklearn.model_selection import train_test_split
from typing import List, Tuple
from variables import class_key, review_key, reviews_class_map
from loguru import logger
import pandas as pd
import tensorflow as tf
from utils import standardize_text, get_precision_recall_fscore

from tensorflow.keras.layers.experimental.preprocessing import TextVectorization

TEST_SIZE = 0.2

# read dataset in batches of
batch_size = 10


def train_test(clean_data: pd.DataFrame) -> Tuple[tf.keras.models.Sequential, tf.keras.models.Sequential]:
    """
    train test
    creates the tensorflow models for sentiment analysis
    """
    logger.info(f'run training and testing for lstm and cnn')

    all_reviews: List[str] = clean_data[review_key]
    labels: List[int] = clean_data[class_key]

    train_reviews, test_reviews, train_labels, test_labels = train_test_split(
        all_reviews, labels, test_size=TEST_SIZE)

    training_dataset = tf.data.Dataset.from_tensor_slices(
        (train_reviews, train_labels))
    testing_dataset = tf.data.Dataset.from_tensor_slices(
        (test_reviews, test_labels))

    # buffer size is used to shuffle the dataset
    buffer_size = 10000
    training_dataset = training_dataset.shuffle(buffer_size).batch(
        batch_size, drop_remainder=True)
    testing_dataset = testing_dataset.shuffle(buffer_size).batch(
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

    output_layer = tf.keras.layers.Dense(1)

    lstm_model = tf.keras.models.Sequential([
        vectorize_layer,
        embedding_layer,
        tf.keras.layers.LSTM(128),
        tf.keras.layers.Dense(64, activation='relu'),
        output_layer,
    ])

    learning_rate: int = 1e-3
    optimizer = tf.keras.optimizers.Adam(learning_rate)
    loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)

    metrics = [tf.keras.metrics.BinaryAccuracy()]

    lstm_model.compile(optimizer=optimizer,
                       loss=loss,
                       metrics=metrics)

    lstm_model.fit(
        training_dataset,
        epochs=1,
        callbacks=[])

    logger.info('lstm model summary:')
    lstm_model.summary()

    _loss, accuracy = lstm_model.evaluate(testing_dataset)
    logger.info(f'accuracy: {accuracy}')

    classes = range(2)

    precision, recall, f_score, support = get_precision_recall_fscore(
        lstm_model, testing_dataset, classes)

    for i in classes:
        logger.info(f'{reviews_class_map[i]}:')
        logger.info(f'precision: {precision[i]}')
        logger.info(f'recall: {recall[i]}')
        logger.info(f'f-score: {f_score[i]}')
        logger.info(f'support: {support[i]}')

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

    cnn_model.fit(training_dataset, epochs=1,
                  callbacks=[])

    logger.info('cnn model summary')
    cnn_model.summary()

    _loss, accuracy = cnn_model.evaluate(testing_dataset)
    logger.info(f'accuracy: {accuracy}')

    precision, recall, f_score, support = get_precision_recall_fscore(
        cnn_model, testing_dataset, classes)

    for i in classes:
        logger.info(f'{reviews_class_map[i]}:')
        logger.info(f'precision: {precision[i]}')
        logger.info(f'recall: {recall[i]}')
        logger.info(f'f-score: {f_score[i]}')
        logger.info(f'support: {support[i]}')

    return lstm_model, cnn_model


if __name__ == '__main__':
    raise ValueError('cannot run classification on its own')
