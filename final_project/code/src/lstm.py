#!/usr/bin/env python3
"""
lstm file

run lstm on dataset
"""

import tensorflow as tf
import numpy as np
from loguru import logger
from utils import roc_auc, plot_train_val_loss


def run_lstm(strategy: tf.distribute.TPUStrategy, x_train_padded: np.array,
             x_valid_padded: np.array, y_train: np.array, y_valid: np.array,
             max_len: int, embedding_size_x: int, embedding_size_y: int,
             embedding_matrix: np.array, epochs: int) -> tf.keras.models.Sequential:
    """
    create and run lstm on training and testing data
    """
    logger.info('build lstm')

    with strategy.scope():
        # A simple LSTM with glove embeddings and one dense layer
        model = tf.keras.models.Sequential()
        model.add(tf.keras.layers.Embedding(embedding_size_x,
                                            embedding_size_y,
                                            weights=[embedding_matrix],
                                            input_length=max_len,
                                            trainable=False))

        model.add(tf.keras.layers.LSTM(
            100, dropout=0.3, recurrent_dropout=0.3))
        model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
        model.compile(loss='binary_crossentropy',
                      optimizer='adam', metrics=['accuracy'])

    model.summary()

    # fit model
    history = model.fit(x_train_padded, y_train, epochs=epochs, batch_size=64 *
                        strategy.num_replicas_in_sync)
    plot_train_val_loss(history, 'lstm')

    scores = model.predict(x_valid_padded)
    logger.info(f"AUC: {roc_auc(scores, y_valid):.4f}")

    return model


if __name__ == '__main__':
    raise RuntimeError('cannot run lstm on its own')
