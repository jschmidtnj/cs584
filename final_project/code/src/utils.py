#!/usr/bin/env python3
"""
utils functions (utils.py)
"""

from os.path import abspath, join
from pathlib import Path
from sklearn.metrics import roc_curve, auc


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
