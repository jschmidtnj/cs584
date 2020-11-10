#!/usr/bin/env python3
"""
utils functions (utils.py)
"""

import re
import string
import numpy as np
from glob import glob
from os.path import abspath, join
from loguru import logger
import tensorflow as tf
import matplotlib.pyplot as plt
from typing import List, Tuple, Optional, Any
from pathlib import Path
from sklearn.metrics import precision_recall_fscore_support
from variables import output_folder

def file_path_relative(rel_path: str) -> str:
    """
    get file path relative to base folder
    """
    return join(
        abspath(join(Path(__file__).absolute(), '../..')), rel_path)


def get_glob(glob_rel_path: str) -> List[str]:
    """
    get glob file list for given path
    """
    logger.info("getting files using glob")
    complete_path: str = file_path_relative(glob_rel_path)
    files = glob(complete_path)
    return files

def standardize_text(text: tf.Tensor):
    lowercase = tf.strings.lower(text)
    without_spaces = tf.strings.strip(lowercase)
    return tf.strings.regex_replace(without_spaces,
                                    '[%s]' % re.escape(string.punctuation),
                                    '')


def get_precision_recall_fscore(model: tf.keras.models.Sequential,
                                dataset: tf.data.Dataset,
                                classes: List[int]) -> Tuple[List[float], List[float], List[float], List[float]]:
    labels: List[float] = []
    predictions: List[float] = []
    for input_batch, label_batch in dataset:
        for i, prediction in enumerate(model.predict(input_batch)):
            max_index = tf.constant(np.argmax(prediction))
            labels.append(label_batch[i].numpy())
            predictions.append(max_index.numpy())

    precision, recall, f_score, support = precision_recall_fscore_support(
        labels, predictions, labels=classes)
    return precision, recall, f_score, support

class UseMaxWeights(tf.keras.callbacks.Callback):
    def __init__(self, patience=0):
        super().__init__()
        self.patience = patience
        self.best_weights: Optional[Any] = None
        self.best: float = np.Inf
        self.wait = 0
        self.stopped_epoch = 0

    def on_train_begin(self, logs=None):
        self.wait = 0
        self.stopped_epoch = 0
        self.best: float = np.Inf

    def on_epoch_end(self, epoch, logs=None):
        current_loss = logs.get("loss")
        if np.less(current_loss, self.best):
            self.best = current_loss
            self.wait = 0
            self.best_weights = self.model.get_weights()
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.stopped_epoch = epoch
                self.model.stop_training = True
                self.model.set_weights(self.best_weights)

    def on_train_end(self, logs=None):
        if self.stopped_epoch > 0:
            logger.info(f"Stopped training early at epoch {self.stopped_epoch + 1}")

def plot_train_val_loss(history: tf.keras.callbacks.History, model_name: str) -> None:

    plt.style.use("seaborn")
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
