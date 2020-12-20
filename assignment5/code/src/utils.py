#!/usr/bin/env python3
"""
utils functions (utils.py)
"""

from os.path import abspath, join
from pathlib import Path
from sklearn.metrics import roc_curve, auc
from variables import raw_data_folder, IN_NOTEBOOK


def file_path_relative(rel_path: str, base_folder: str = raw_data_folder) -> str:
    """
    get file path relative to base folder
    """
    if IN_NOTEBOOK:
        current_path = './'
    else:
        current_path = join(Path(__file__).absolute(), '../../')
    return join(abspath(current_path), base_folder, rel_path)


def roc_auc(predictions, target):
    """
    This methods returns the AUC Score when given the Predictions
    and Labels
    """

    fpr, tpr, _thresholds = roc_curve(target, predictions)
    return auc(fpr, tpr)
