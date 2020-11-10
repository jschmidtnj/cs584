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
from typing import List, Tuple
from pathlib import Path
from sklearn.metrics import precision_recall_fscore_support

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
