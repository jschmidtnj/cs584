#!/usr/bin/env python3
"""
roberta file

run roberta on dataset
"""

import tensorflow as tf
import numpy as np
from loguru import logger
from utils import roc_auc, build_model
from transformers import TFRobertaPreTrainedModel


def run_roberta(strategy: tf.distribute.TPUStrategy, x_train: np.array,
                x_valid: np.array, _y_train: np.array, y_valid: np.array,
                train_dataset: tf.data.Dataset, valid_dataset: tf.data.Dataset,
                test_dataset: tf.data.Dataset, max_len: int, epochs: int,
                batch_size: int) -> tf.keras.models.Model:
    """
    create and run distilibert on training and testing data
    """
    logger.info('build roberta')

    with strategy.scope():
        transformer_layer = TFRobertaPreTrainedModel()
        model = build_model(transformer_layer, max_len=max_len)
    model.summary()

    n_steps = x_train.shape[0] // batch_size
    _train_history = model.fit(
        train_dataset,
        steps_per_epoch=n_steps,
        validation_data=valid_dataset,
        epochs=epochs
    )

    n_steps = x_valid.shape[0] // batch_size
    _train_history_2 = model.fit(
        valid_dataset.repeat(),
        steps_per_epoch=n_steps,
        epochs=epochs
    )

    scores = model.predict(test_dataset, verbose=1)
    logger.info(f"AUC: {roc_auc(scores, y_valid):.2f}")

    return model


if __name__ == '__main__':
    raise RuntimeError('cannot run roberta on its own')
