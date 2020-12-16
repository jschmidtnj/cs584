#!/usr/bin/env python3
"""
utils functions (utils.py)
"""

from os.path import abspath, join
from pathlib import Path
from sklearn.metrics import roc_curve, auc
from typing import Any
import tensorflow as tf
import matplotlib.pyplot as plt
from variables import raw_data_folder, IN_KAGGLE, output_folder

default_base_folder: str = raw_data_folder if not IN_KAGGLE else 'jigsaw-multilingual-toxic-comment-classification'


def file_path_relative(rel_path: str, base_folder: str = default_base_folder) -> str:
    """
    get file path relative to base folder
    """
    current_path = join(Path(__file__).absolute(), '../../')
    if IN_KAGGLE:
        current_path = '/kaggle/input/'
    return join(abspath(current_path), base_folder, rel_path)


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


def plot_train_val_loss(history: tf.keras.callbacks.History, model_name: str) -> None:
    """
    plots the training and validation loss given training history
    """

    plt.figure()

    loss_train = history.history['loss']
    loss_val = history.history['val_loss']

    num_epochs = len(loss_train)
    nums = range(1, num_epochs + 1)

    plt.plot(nums, loss_train, label="train")
    plt.plot(nums, loss_val, label="validation")
    plt.title(f"Training and Validation loss over {num_epochs} epochs")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss")
    plt.legend()
    file_path = file_path_relative(
        f'{output_folder}/{model_name}.png')
    plt.savefig(file_path)
