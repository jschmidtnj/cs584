#!/usr/bin/env python3
"""
rnn file

run rnn on dataset
"""

import tensorflow as tf
import numpy as np
from loguru import logger
from utils import roc_auc


def run_rnn(strategy: tf.distribute.TPUStrategy, x_train_padded: np.array,
            x_valid_padded: np.array, y_train: np.array, y_valid: np.array,
            max_len: int, embedding_size_x: int, embedding_size_y: int,
            embedding_matrix: np.array) -> tf.keras.models.Sequential:
    """
    create and run bidirectional rnn on training and testing data
    """
    logger.info('build rnn')

    with strategy.scope():
        # A simple bidirectional LSTM with glove embeddings and one dense layer
        model = tf.keras.models.Sequential()
        model.add(tf.keras.layers.Embedding(embedding_size_x,
                                            embedding_size_y,
                                            weights=[embedding_matrix],
                                            input_length=max_len,
                                            trainable=False))
        model.add(tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(embedding_size_y, dropout=0.3, recurrent_dropout=0.3)))

        model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
        model.compile(loss='binary_crossentropy',
                      optimizer='adam', metrics=['accuracy'])

    model.summary()

    model.fit(x_train_padded, y_train, batch_size=64*strategy.num_replicas_in_sync)

    scores = model.predict(x_valid_padded)
    logger.info(f"AUC: {roc_auc(scores, y_valid):.4f}")

    return model


if __name__ == '__main__':
    raise RuntimeError('cannot run rnn on its own')
