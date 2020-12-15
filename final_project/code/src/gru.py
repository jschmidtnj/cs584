#!/usr/bin/env python3
"""
gru file

run gru on dataset
"""

import tensorflow as tf
import numpy as np
from loguru import logger
from utils import roc_auc


def run_gru(strategy: tf.distribute.TPUStrategy, x_train_padded: np.array,
            x_valid_padded: np.array, y_train: np.array, y_valid: np.array,
            max_len: int, embedding_size_x: int, embedding_size_y: int,
            embedding_matrix: np.array) -> tf.keras.models.Sequential:
    """
    create and run gru on training and testing data
    """
    logger.info('build gru')

    with strategy.scope():
        # GRU with glove embeddings and two dense layers
        model = tf.keras.models.Sequential()
        model.add(tf.keras.layers.Embedding(embedding_size_x,
                                            embedding_size_y,
                                            weights=[embedding_matrix],
                                            input_length=max_len,
                                            trainable=False))
        model.add(tf.keras.layers.SpatialDropout1D(0.3))
        model.add(tf.keras.layers.GRU(embedding_size_y))
        model.add(tf.keras.layers.Dense(1, activation='sigmoid'))

        model.compile(loss='binary_crossentropy',
                      optimizer='adam', metrics=['accuracy'])

    model.summary()

    model.fit(x_train_padded, y_train, nb_epoch=5,
              batch_size=64*strategy.num_replicas_in_sync)

    scores = model.predict(x_valid_padded)
    logger.info(f"AUC: {roc_auc(scores, y_valid):.2f}")

    return model


if __name__ == '__main__':
    raise RuntimeError('cannot run gru on its own')
