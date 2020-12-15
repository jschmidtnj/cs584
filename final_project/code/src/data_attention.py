#!/usr/bin/env python3
"""
data file

read in data
"""

from typing import Tuple
import pandas as pd
import tensorflow as tf
from loguru import logger
from utils import file_path_relative
from variables import raw_data_folder
from tqdm import tqdm
import numpy as np
from transformers import DistilBertTokenizer
from tokenizers import BertWordPieceTokenizer

NUM_ROWS_TRAIN: int = 15000
TEST_RATIO: float = 0.2


def _fast_encode(texts: np.array, tokenizer: BertWordPieceTokenizer, chunk_size: int = 256, maxlen: int = 512):
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


def read_data_attention(strategy: tf.distribute.TPUStrategy,
                        max_len: int) -> Tuple[np.array, np.array, np.array, np.array, tf.data.Dataset, tf.data.Dataset, tf.data.Dataset, int]:
    """
    read data from attention models
    """
    logger.info('reading data for attention models')

    batch_size = 16 * strategy.num_replicas_in_sync
    auto = tf.data.experimental.AUTOTUNE

    # First load the real tokenizer
    tokenizer = DistilBertTokenizer.from_pretrained(
        'distilbert-base-multilingual-cased')
    # Save the loaded tokenizer locally
    tokenizer.save_pretrained('.')
    # Reload it with the huggingface tokenizers library
    fast_tokenizer = BertWordPieceTokenizer('vocab.txt', lowercase=False)

    train = pd.read_csv(file_path_relative(
        f'{raw_data_folder}/jigsaw-toxic-comment-train.csv'))
    valid = pd.read_csv(file_path_relative(
        f'{raw_data_folder}/validation.csv'))
    test = pd.read_csv(file_path_relative(f'{raw_data_folder}/test.csv'))

    x_train = _fast_encode(train['comment_text'].astype(str),
                           fast_tokenizer, maxlen=max_len)
    x_valid = _fast_encode(valid['comment_text'].astype(str),
                           fast_tokenizer, maxlen=max_len)
    x_test = _fast_encode(test['content'].astype(
        str), fast_tokenizer, maxlen=max_len)

    y_train = train.toxic.values
    y_valid = valid.toxic.values

    train_dataset = (
        tf.data.Dataset
        .from_tensor_slices((x_train, y_train))
        .repeat()
        .shuffle(2048)
        .batch(batch_size)
        .prefetch(auto)
    )

    valid_dataset = (
        tf.data.Dataset
        .from_tensor_slices((x_valid, y_valid))
        .batch(batch_size)
        .cache()
        .prefetch(auto)
    )

    test_dataset = (
        tf.data.Dataset
        .from_tensor_slices(x_test)
        .batch(batch_size)
    )

    return x_train, x_valid, y_train, y_valid, train_dataset, valid_dataset, test_dataset, batch_size


if __name__ == '__main__':
    raise RuntimeError('cannot run data attention on its own')
