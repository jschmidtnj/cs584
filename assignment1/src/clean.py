#!/usr/bin/env python3
"""
data clean (clean.py)
"""

from os.path import basename, splitext
from typing import Optional, List
from utils import get_glob
from variables import data_folder, paragraph_key, label_key, random_state, class_key
from loguru import logger
from books import BookType, start_end_map, class_map
import pandas as pd
from typing import Tuple

title_split: str = 'title: '
author_split: str = 'author: '

start_book: str = 'start of this project gutenberg ebook'
the_end: str = 'the end'
end_book: str = 'end of this project gutenberg ebook'

chapter: str = 'Chapter '
adventure: str = 'ADVENTURE '
multi_quote_identifier: str = '"'

min_line_len: int = 6  # line discarded if less than this number of characters


def clean() -> Tuple[pd.DataFrame, List[BookType]]:
    """
    data cleaning
    """
    data: pd.DataFrame = pd.DataFrame()
    class_count: int = 0
    label_list: List[BookType] = []

    # preprocess data and construct examples
    for file_path in get_glob(f'{data_folder}/*.txt'):
        file_name: str = basename(splitext(file_path)[0])
        logger.info(f'processing {file_name}')
        title: Optional[str] = None
        book_key: Optional[BookType] = None
        book_started: bool = False
        paragraphs: List[List[str]] = []
        num_newline_count: int = 0
        line_number: int = 0
        with open(file_path, 'r') as current_file:
            while True:
                line = current_file.readline()
                line_number += 1
                line_lower_trim: Optional[str] = None
                if line:
                    line_lower_trim = line.lower().strip()
                if not book_started and \
                    ((line_lower_trim is not None and line_lower_trim.startswith(start_book))
                     or (book_key is not None and line_number >= start_end_map[book_key].start)):
                    book_started = True
                if line_lower_trim is None or line_lower_trim.startswith(end_book) \
                        or line_lower_trim == the_end or \
                        (book_key is not None and line_number >= start_end_map[book_key].end):
                    # done with reading the file
                    break
                if not book_started:
                    if title is None and line_lower_trim.startswith(title_split):
                        title = line_lower_trim.split(title_split)[1]
                        logger.info(f'title: {title}')
                    if book_key is None and line_lower_trim.startswith(author_split):
                        author: str = line_lower_trim.split(author_split)[1]
                        logger.info(f'author: {author}')
                        book_key = BookType(author.split(' ')[-1])
                else:
                    if len(line_lower_trim) < min_line_len or \
                            line.startswith(chapter) or line.startswith(chapter):
                        num_newline_count += 1
                    else:
                        multi_line_quotes = line_lower_trim.startswith(multi_quote_identifier) \
                            and paragraphs[-1][0].startswith(multi_quote_identifier)
                        if len(paragraphs) == 0 or \
                                (num_newline_count > 0 and not multi_line_quotes):
                            paragraphs.append([])
                        num_newline_count = 0
                        paragraphs[-1].append(line_lower_trim)
        if book_key is None:
            raise RuntimeError('no book key found')
        class_name = class_map[book_key]
        logger.info(
            f'number of paragraphs in class "{class_name}": {len(paragraphs)}')
        data = pd.concat([data, pd.DataFrame({
            paragraph_key: [' '.join(paragraph) for paragraph in paragraphs],
            label_key: [class_name] * len(paragraphs),
            class_key: class_count
        })], ignore_index=True)
        label_list.append(book_key)
        class_count += 1
    logger.info(
        f'\nsample of output data:\n{data.sample(random_state=random_state, n=5)}')
    return data, label_list


if __name__ == '__main__':
    clean()
