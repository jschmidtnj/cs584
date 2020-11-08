#!/usr/bin/env python3
"""
data clean (clean.py)
"""

import re
from ast import literal_eval
from os.path import basename, splitext, exists
from typing import Optional, List
from utils import get_glob, file_path_relative
from variables import part_1_data_folder, clean_data_folder, class_key, label_key, paragraph_key
from loguru import logger
from books import BookType, start_end_map, class_map
import pandas as pd
from typing import Tuple
import yaml

title_split: str = 'title: '
author_split: str = 'author: '

start_book: str = 'start of this project gutenberg ebook'
the_end: str = 'the end'
end_book: str = 'end of this project gutenberg ebook'

chapter: str = 'Chapter '
adventure: str = 'ADVENTURE '
multi_quote_identifier: str = '"'

min_line_len: int = 6  # line discarded if less than this number of characters

default_file_name: str = f'{clean_data_folder}/documents.csv'
classes_file_name: str = f'{clean_data_folder}/doc_classes.txt'

whitespace_regex = re.compile(r"\s+")

def normalize_sentence(sentence: str) -> str:
    """
    remove punctuation, return list of words
    """
    sentence = whitespace_regex.sub(' ', sentence).strip()
    return sentence


def clean(clean_data_basename: Optional[str] = default_file_name) -> Tuple[pd.DataFrame, List[BookType]]:
    """
    data cleaning
    """
    class_count: int = 0
    label_list: List[BookType] = []

    get_from_disk = clean_data_basename is not None

    if not get_from_disk:
        clean_data_basename = default_file_name

    clean_data_path = file_path_relative(clean_data_basename)
    classes_path = file_path_relative(classes_file_name)

    if get_from_disk and exists(clean_data_path) and exists(classes_path):
        logger.info(f'reading data from {clean_data_path}')
        data = pd.read_csv(clean_data_path, converters={
            paragraph_key: literal_eval})
        label_list_enum: Optional[List[BookType]] = None
        with open(classes_path) as classes_file:
            label_list = yaml.load(classes_file, Loader=yaml.FullLoader)
            label_list_enum = [BookType(elem) for elem in label_list]
        return data, label_list_enum

    data: pd.DataFrame = pd.DataFrame()

    # preprocess data and construct examples
    found_files: bool = False
    for file_path in get_glob(f'{part_1_data_folder}/*.txt'):
        found_files = True
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
                line_trim: Optional[str] = None
                if line:
                    line_trim = line.strip()
                if not book_started and \
                    ((line_trim is not None and line_trim.startswith(start_book))
                     or (book_key is not None and line_number >= start_end_map[book_key].start)):
                    book_started = True
                if line_trim is None or line_trim.startswith(end_book) \
                        or line_trim == the_end or \
                        (book_key is not None and line_number >= start_end_map[book_key].end):
                    # done with reading the file
                    break
                if not book_started:
                    if title is None and line_trim.startswith(title_split):
                        title = line_trim.split(title_split)[1]
                        logger.info(f'title: {title}')
                    if book_key is None and line_trim.startswith(author_split):
                        author: str = line_trim.split(author_split)[1]
                        logger.info(f'author: {author}')
                        book_key = BookType(author.split(' ')[-1])
                else:
                    if len(line_trim) < min_line_len or \
                            line.startswith(chapter) or line.startswith(chapter):
                        num_newline_count += 1
                    else:
                        multi_line_quotes = line_trim.startswith(multi_quote_identifier) \
                            and paragraphs[-1][0].startswith(multi_quote_identifier)
                        if len(paragraphs) == 0 or \
                                (num_newline_count > 0 and not multi_line_quotes):
                            paragraphs.append([])
                        num_newline_count = 0
                        paragraphs[-1].append(line_trim)
        if not found_files:
            raise RuntimeError('no files found')
        if book_key is None:
            raise RuntimeError('no book key found')
        class_name = class_map[book_key]
        logger.info(
            f'number of paragraphs in class "{class_name}": {len(paragraphs)}')
        paragraphs = [[normalize_sentence(sentence) for sentence in paragraph] for paragraph in paragraphs]
        data = pd.concat([data, pd.DataFrame({
            paragraph_key: paragraphs,
            label_key: [class_name] * len(paragraphs),
            class_key: class_count
        })], ignore_index=True)
        label_list.append(book_key)
        class_count += 1

    data.to_csv(clean_data_path, index=False)
    with open(classes_path, 'w') as classes_file:
        label_list_str = [elem.name for elem in label_list]
        yaml.dump(label_list_str, classes_file)

    return data, label_list


if __name__ == '__main__':
    clean()
