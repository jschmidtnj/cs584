#!/usr/bin/env python3
"""
distilibert file

run distilibert on dataset
"""

import tensorflow as tf
import numpy as np
from loguru import logger
from utils import roc_auc, build_model, plot_train_val_loss
from transformers import TFDistilBertModel

MODEL: str = 'distilbert-base-multilingual-cased'


def run_distilibert(strategy: tf.distribute.TPUStrategy, x_train: np.array,
                    x_valid: np.array, _y_train: np.array, y_valid: np.array,
                    train_dataset: tf.data.Dataset, valid_dataset: tf.data.Dataset,
                    test_dataset: tf.data.Dataset, max_len: int, epochs: int,
                    batch_size: int) -> tf.keras.models.Model:
    """
    create and run distilbert on training and testing data
    """
    logger.info('build distilbert')

    with strategy.scope():
        transformer_layer = TFDistilBertModel.from_pretrained(MODEL)
        model = build_model(transformer_layer, max_len=max_len)
    model.summary()

    # train given model
    n_steps = x_train.shape[0] // batch_size
    history = model.fit(
        train_dataset,
        steps_per_epoch=n_steps,
        validation_data=valid_dataset,
        epochs=epochs
    )
    plot_train_val_loss(history, 'distilbert')

    n_steps = x_valid.shape[0] // batch_size
    _train_history_2 = model.fit(
        valid_dataset.repeat(),
        steps_per_epoch=n_steps,
        epochs=epochs*2
    )

    scores = model.predict(test_dataset, verbose=1)
    logger.info(f"AUC: {roc_auc(scores, y_valid):.4f}")

    return model


if __name__ == '__main__':
    raise RuntimeError('cannot run distilibert on its own')
