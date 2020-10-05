#!/usr/bin/env python
"""
KNN model
"""

import numpy as np
from typing import List

def _get_distance_euclidian(row1: np.array, row2: np.array):
    """
    _get_distance
    returns the distance between 2 rows
    (euclidian distance between vectors)
    takes into account all columns of data given
    """
    distance = 0.
    for i, _ in enumerate(row1):
        distance += (row1[i] - row2[i]) ** 2
    return np.sqrt(distance)

def _get_distance_cosine(row1: np.array, row2: np.array) -> float:
    """
    _get_distance
    returns the distance between 2 rows
    (cosine similarity between vectors)
    takes into account all columns of data given
    """
    return np.dot(row1, row2) / (np.linalg.norm(row1) * np.linalg.norm(row2))

equal_to_dist: float = 1e-8

def run_knn(vector: np.array, matrix: np.array, k: int, tokens) -> np.array:
    """
    run_knn
    gets k neighbors, by first getting all the distances,
    sorting them, and returning the indexes associated with
    the k closest neighbors
    """
    distances = []
    for token in tokens:
        current_index = tokens[token]
        training_row = matrix[current_index]
        dist = _get_distance_cosine(training_row, vector)
        if abs(1 - dist) > equal_to_dist:
            distances.append((dist, current_index))
    distances = sorted(distances, key=lambda x: x[0], reverse=True)
    neighbor_indexes = []
    for i in range(k):
        neighbor_indexes.append(distances[i][1])
    return np.array(neighbor_indexes, dtype=np.int)
