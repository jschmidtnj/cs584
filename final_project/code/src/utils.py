#!/usr/bin/env python3
"""
utils functions (utils.py)
"""

from os.path import abspath, join
from pathlib import Path
from sklearn.metrics import roc_curve, auc
from typing import Any
import tensorflow as tf


def file_path_relative(rel_path: str) -> str:
    """
    get file path relative to base folder
    """
    return join(
        abspath(join(Path(__file__).absolute(), '../..')), rel_path)


def roc_auc(predictions, target):
    """
    This methods returns the AUC Score when given the Predictions
    and Labels
    """

    fpr, tpr, _thresholds = roc_curve(target, predictions)
    return auc(fpr, tpr)


def build_model(transformer: Any, max_len: int) -> tf.keras.Model:
    """
    function for building a model given a transformer and max length
    """
    input_word_ids = tf.keras.Input(
        shape=(max_len,), dtype=tf.int32, name="input_word_ids")
    sequence_output = transformer(input_word_ids)[0]
    cls_token = sequence_output[:, 0, :]
    out = tf.keras.layers.Dense(1, activation='sigmoid')(cls_token)

    model = tf.keras.Model(inputs=input_word_ids, outputs=out)
    model.compile(tf.keras.optimizers.Adam(lr=1e-5), loss='binary_crossentropy',
                  metrics=['accuracy'])

    return model
