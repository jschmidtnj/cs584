#!/usr/bin/env python3
"""
build embeddings

build embeddings for all data
"""

import numpy as np
from loguru import logger
from utils import file_path_relative, default_base_folder
from variables import IN_KAGGLE
from typing import Dict
from tqdm import tqdm


def build_embeddings(embedding_size_y: int, word_indexes: Dict[str, int]) -> np.array:
    """
    build embeddings to be used with models
    """
    logger.info('build glove embeddings')

    embeddings_indexes: Dict[str, np.array] = {}
    with open(file_path_relative(f'glove.840B.{embedding_size_y}d.txt',
                                 base_folder=default_base_folder if not IN_KAGGLE else 'glove840b300dtxt'),
              encoding='utf-8') as glove_file:
        for line in tqdm(glove_file):
            words = line.split(' ')
            word = words[0]
            coefficients = np.asarray([float(val) for val in words[1:]])
            embeddings_indexes[word] = coefficients

    logger.info(f'Found {len(embeddings_indexes)} word vectors.')

    embedding_size_x: int = len(word_indexes) + 1

    embeddings_output = np.zeros((embedding_size_x, embedding_size_y))
    for word, i in tqdm(word_indexes.items()):
        word_embedding = embeddings_indexes.get(word)
        if word_embedding is not None:
            embeddings_output[i] = word_embedding

    return embeddings_output


if __name__ == '__main__':
    raise RuntimeError('cannot run embeddings on its own')
