#!/usr/bin/env python3
"""
data file

read in data
"""

import tensorflow as tf
from loguru import logger
from typing import List, Tuple, Optional
from sklearn.model_selection import train_test_split
import numpy as np
from os.path import join, dirname
import re
import unicodedata


NUM_ROWS_TRAIN: int = 30000
TEST_RATIO: float = 0.2


def preprocess_without_tokens(sentence: str) -> List[str]:
    """
    preprocess the given sentence, do not add tokens
    """
    # convert to ascii
    sentence = ''.join(c for c in unicodedata.normalize('NFD', sentence.lower().strip())
                       if unicodedata.category(c) != 'Mn')

    # from https://stackoverflow.com/a/3645946/8623391
    sentence = re.sub(r"([?.!,¿])", r" \1 ", sentence)
    sentence = re.sub(r'[" "]+', " ", sentence)

    # remove all invalid characters
    sentence = re.sub(r"[^a-zA-Z?.!,¿]+", " ", sentence)

    sentence = sentence.strip()
    return sentence.split(' ')


def preprocess_sentence(sentence: str) -> str:
    """
    preprocess with tokens
    """
    without_tokens = preprocess_without_tokens(sentence)
    without_tokens.insert(0, '<start>')
    without_tokens.append('<end>')
    return ' '.join(without_tokens)


def get_lines(file_path):
    """
    get all lines in file
    """
    with open(file_path, encoding='utf-8') as data_file:
        lines = map(lambda line: line.strip(), data_file.readlines())
    return lines


def create_dataset(input_file: str, target_file: str, num_examples=None):
    """
    create dataset of lines sorted by length
    """
    input_lines = get_lines(input_file)
    target_lines = get_lines(target_file)
    logger.info('got lines')
    all_lines = list(zip(input_lines, target_lines))
    logger.info('zipped')
    all_lines.sort(key=lambda lines: len(lines[0]))
    logger.info('sorted')
    truncated_lines = all_lines[:num_examples]
    logger.info('truncated')
    unzipped_dataset = tuple(zip(*truncated_lines))
    logger.info('unzipped')
    return unzipped_dataset


def tokenize(data: List[str]) -> Tuple[np.array, tf.keras.preprocessing.text.Tokenizer]:
    """
    tokenize given input
    """
    lang_tokenizer = tf.keras.preprocessing.text.Tokenizer(
        filters='')
    lang_tokenizer.fit_on_texts(data)

    tensor = lang_tokenizer.texts_to_sequences(data)

    tensor = tf.keras.preprocessing.sequence.pad_sequences(tensor,
                                                           padding='post')

    return tensor, lang_tokenizer


def build_tensors(training: List[str], targets: List[str]):
    """
    build training and target tensors
    """
    training_data = [preprocess_sentence(line) for line in training]
    target_data = [preprocess_sentence(line) for line in targets]

    input_tensors, input_language_tokenizer = tokenize(training_data)
    target_tensors, target_language_tokenizer = tokenize(target_data)

    return input_tensors, target_tensors, input_language_tokenizer, target_language_tokenizer


def read_data(dataset_size: Optional[int]) -> Tuple[np.array, np.array, tf.keras.preprocessing.text.Tokenizer, tf.keras.preprocessing.text.Tokenizer, int, int, List[str], List[str]]:
    """
    read data from raw data, convert to dataframes
    """
    logger.info('reading data')

    # Download the file
    path_to_zip = tf.keras.utils.get_file(
        'es-en.tgz', origin='http://www.statmt.org/europarl/v7/es-en.tgz',
        extract=True)

    english_file = join(dirname(path_to_zip), "europarl-v7.es-en.en")
    spanish_file = join(dirname(path_to_zip), "europarl-v7.es-en.es")

    input_dataset, target_dataset = create_dataset(
        spanish_file, english_file, dataset_size)

    # create training and validation data using train-test split
    input_train, input_val, target_train, target_val = train_test_split(
        input_dataset, target_dataset, test_size=TEST_RATIO)
    input_tensor_train, target_tensor_train, inp_lang, targ_lang = build_tensors(
        input_train, target_train)

    # Calculate max_length of the target tensors
    max_length_targ, max_length_inp = target_tensor_train.shape[1], input_tensor_train.shape[1]

    # print length
    logger.info(f'{len(input_tensor_train)}, {len(target_tensor_train)}')

    return input_tensor_train, target_tensor_train, inp_lang, targ_lang, max_length_targ, max_length_inp, input_val, target_val


if __name__ == '__main__':
    raise RuntimeError('cannot run data on its own')
