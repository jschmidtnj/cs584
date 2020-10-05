#!/usr/bin/env python
"""
utility functions
"""

import numpy as np
from glob import glob
from os.path import abspath, join
from loguru import logger
from typing import List, Union
from pathlib import Path

epsilon = 1e-25

def normalizeRows(x: Union[np.array, List[List[float]]]) -> np.array:
    """ Row normalization function
    Implement a function that normalizes each row of a matrix to have
    unit length.

    Arguments:
    x -- A D dimensional vector or N x D dimensional numpy matrix.
    Return:
    x -- You are allowed to modify x in-place
    """
    x = np.array(x, dtype=np.double)
    x_norm = np.sqrt(np.sum(x**2, axis=1))
    x_norm = x_norm.reshape((x.shape[0], 1)) # ensure the shape is the same
    x_norm += epsilon # prevent divide by zero
    x = x / x_norm
    return x


def softmax(x: Union[np.array, List[List[float]]]) -> np.array:
    """Compute the softmax function for each row of the input x.
    It is crucial that this function is optimized for speed because
    it will be used frequently in later code. 

    Arguments:
    x -- A D dimensional vector or N x D dimensional numpy matrix.
    Return:
    x -- You are allowed to modify x in-place
    """
    # first cast to numpy array
    x = np.array(x, dtype=np.double)
    shape = x.shape

    if len(shape) == 1:
        # single-dimensional array
        x = np.exp(x - np.max(x))
        x = x / np.sum(x)
    else:
        # multi-dimensional array - needs to be reshaped
        new_shape = (x.shape[0], 1)
        x = np.exp(x - np.max(x, axis=1).reshape(new_shape))
        x = x / np.sum(x, axis=1).reshape(new_shape)

    return x

def get_relative_path(rel_path: str) -> str:
    """
    get abs file path for given relative path
    """
    complete_path: str = join(
        abspath(join(Path(__file__).absolute(), '../../..')), rel_path)
    return complete_path
