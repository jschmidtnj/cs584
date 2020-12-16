#!/usr/bin/env python3
"""
data file

read in data
"""

import pandas as pd
import tensorflow as tf
from loguru import logger
from utils import file_path_relative
from typing import List, Tuple, Dict
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import numpy as np


def fast_encode(texts, tokenizer, chunk_size=256, maxlen=512):
    """
    Encoder for encoding the text into sequence of integers for BERT Input
    """
    tokenizer.enable_truncation(max_length=maxlen)
    tokenizer.enable_padding(max_length=maxlen)
    all_ids = []

    for i in tqdm(range(0, len(texts), chunk_size)):
        text_chunk = texts[i:i+chunk_size].tolist()
        encs = tokenizer.encode_batch(text_chunk)
        all_ids.extend([enc.ids for enc in encs])

    return np.array(all_ids)


NUM_ROWS_TRAIN: int = 15000
TEST_RATIO: float = 0.2


def read_data() -> Tuple[np.array, np.array, np.array, np.array, int, Dict[str, int]]:
    """
    read data from raw data, convert to dataframes
    """
    logger.info('reading data')

    train = pd.read_csv(file_path_relative('jigsaw-toxic-comment-train.csv'))

    # drop unused columns
    train.drop(['severe_toxic', 'obscene', 'threat', 'insult',
                'identity_hate'], axis=1, inplace=True)

    # only use first n rows
    train = train.loc[:NUM_ROWS_TRAIN, :]
    logger.info(f'shape of training data: {train.shape}')

    max_len = train['comment_text'].apply(
        lambda x: len(str(x).split())).max()
    logger.info(f'max len: {max_len}')

    x_train, x_valid, y_train, y_valid = train_test_split(train['comment_text'].values, train['toxic'].values,
                                                          stratify=train['toxic'].values,
                                                          test_size=TEST_RATIO, shuffle=True)

    tokens = tf.keras.preprocessing.text.Tokenizer(num_words=None)

    all_data: List[str] = list(x_train)
    all_data.extend(list(x_valid))
    tokens.fit_on_texts(all_data)
    x_train_sequences = tokens.texts_to_sequences(x_train)
    x_valid_sequences = tokens.texts_to_sequences(x_valid)

    # pad the data with zeros
    x_train_padded = tf.keras.preprocessing.sequence.pad_sequences(
        x_train_sequences, maxlen=max_len)
    x_valid_padded = tf.keras.preprocessing.sequence.pad_sequences(
        x_valid_sequences, maxlen=max_len)

    word_indexes = tokens.word_index

    return x_train_padded, x_valid_padded, y_train, y_valid, max_len, word_indexes


if __name__ == '__main__':
    raise RuntimeError('cannot run data on its own')
