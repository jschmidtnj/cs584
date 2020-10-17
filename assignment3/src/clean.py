#!/usr/bin/env python3
"""
data clean (clean.py)
"""

from os.path import basename, splitext, join
from typing import List, Optional
from utils import get_glob
from variables import raw_data_folder, clean_data_folder, sentences_key, \
    random_state, unknown_token
from loguru import logger
import pandas as pd
import re

# these are contractions that don't start with an apostrophe, but should
# still be joined with the previous word
additional_contractions: List[str] = set(["n't"])

# these are characters that should be removed from the tokens
unwanted_chars: List[str] = [',']

# these are words that should not be in the corpus
unwanted_words: List[str] = ['___']

# unknown tokens are tokens in the corpus that are mapped to an unknown token
unknown_token_list: List[str] = ['<unk>']


def clean_tokenize(file_name: Optional[str] = None) -> List[pd.DataFrame]:
    """
    data cleaning
    create array of array sentences
    tokens are words
    """
    if file_name is None:
        file_name = '*.txt'

    unwanted_chars_regex = f"[{''.join(unwanted_chars)}]"

    res: List[pd.DataFrame] = []

    # preprocess data
    for file_path in get_glob(f'{raw_data_folder}/{file_name}'):
        file_name: str = basename(splitext(file_path)[0])
        logger.info(f'processing {file_name}')
        sentences: List[List[str]] = []
        line_number: int = 0
        with open(file_path, 'r') as current_file:
            while True:
                line: Optional[str] = current_file.readline()
                if not line:
                    break
                line_number += 1
                sentence: List[float] = []
                line_lower_trim = line.lower().strip()
                line_sanitized = re.sub(
                    unwanted_chars_regex, '', line_lower_trim)
                for word in line_sanitized.split(' '):
                    if len(sentence) > 0 and (word.startswith("'")
                                              or word in additional_contractions):
                        sentence[-1] += word
                    else:
                        if word in unknown_token_list:
                            word = unknown_token
                        sentence.append(word)
                sentences.append(sentence)
        logger.info(f'read {line_number} lines from {file_name}')
        data: pd.DataFrame = pd.DataFrame({
            sentences_key: sentences
        })
        data.to_csv(join(clean_data_folder, f'{file_name}.csv'))
        logger.info(
            f'\nsample of clean data for {file_name}: \n{data.sample(random_state=random_state, n=5)}')
        res.append(data)

    return res


if __name__ == '__main__':
    clean_tokenize()
