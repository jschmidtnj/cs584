#!/usr/bin/env python3
"""
simple rnn file

run simple rnn on dataset
"""

import tensorflow as tf
import numpy as np
from loguru import logger
from utils import roc_auc, plot_train_val_loss


def simple_rnn(strategy: tf.distribute.TPUStrategy, x_train_padded: np.array,
               x_valid_padded: np.array, y_train: np.array, y_valid: np.array,
               max_len: int, embedding_size_x: int, embedding_size_y: int, epochs: int) -> tf.keras.models.Sequential:
    """
    create and run simple rnn on training and testing data
    """
    logger.info('build simple RNN')

    with strategy.scope():
        # A simpleRNN without any pretrained embeddings and one dense layer
        model = tf.keras.models.Sequential()
        model.add(tf.keras.layers.Embedding(embedding_size_x, embedding_size_y,
                                            input_length=max_len))
        model.add(tf.keras.layers.SimpleRNN(100))
        model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
        model.compile(loss='binary_crossentropy',
                      optimizer='adam', metrics=['accuracy'])

    model.summary()

    # run model train
    history = model.fit(x_train_padded, y_train, epochs=epochs, batch_size=64 *
                        strategy.num_replicas_in_sync)
    plot_train_val_loss(history, 'simple_rnn')

    scores = model.predict(x_valid_padded)
    logger.info(f"AUC: {roc_auc(scores, y_valid):.4f}")

    return model


if __name__ == '__main__':
    raise RuntimeError('cannot run simple rnn on its own')
