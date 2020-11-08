#!/usr/bin/env python3
"""
clean reviews (clean_reviews.py)
"""

import re
from lxml import etree
from ast import literal_eval
from os.path import basename, splitext, exists
from typing import Optional, List, cast
from utils import get_glob, file_path_relative
from variables import part_2_data_folder, clean_data_folder, class_key, label_key, review_key
from loguru import logger
from books import BookType, start_end_map, class_map
import pandas as pd
from typing import Tuple
import yaml

default_file_name: str = f'{clean_data_folder}/reviews.csv'

whitespace_regex = re.compile(r"\s+")

def normalize_review(review: str) -> str:
    """
    remove punctuation, return list of words
    """
    review = whitespace_regex.sub(' ', review).strip()
    return review


def clean(clean_data_basename: Optional[str] = default_file_name) -> pd.DataFrame:
    """
    data cleaning
    """
    data: pd.DataFrame = pd.DataFrame()
    class_count: int = 0
    label_list: List[BookType] = []

    get_from_disk = clean_data_basename is not None

    if not get_from_disk:
        clean_data_basename = default_file_name

    clean_data_path = file_path_relative(clean_data_basename)

    if get_from_disk and exists(clean_data_path):
        logger.info(f'reading data from {clean_data_path}')
        data = pd.read_csv(clean_data_path)
        return data

    data: pd.DataFrame = pd.DataFrame()

    for class_val, file_path in enumerate([file_path_relative(f'{part_2_data_folder}/negative.review'),
                                file_path_relative(f'{part_2_data_folder}/positive.review')]):
        root: Optional[etree._Element] = None
        with open(file_path, 'rb') as current_file:
            parser = etree.XMLParser(recover=True)
            root = etree.fromstring(f'<?xml version="1.0"?><root_elem>{current_file.read()}</root_elem>', parser=parser)

        reviews: List[str] = []

        for elem in root.findall('.//review_text'):
            cast_elem: etree._Element = cast(etree._Element, elem)
            decoded_text: str = literal_eval(f"'{cast_elem.text}'")
            trimmed_text = decoded_text.strip()
            reviews.append(trimmed_text)

        class_name: str = 'Negative' if class_val == 0 else 'Positive'

        logger.info(
            f'number of reviews in class "{class_name}": {len(reviews)}')

        data = pd.concat([data, pd.DataFrame({
            review_key: reviews,
            label_key: class_name,
            class_key: class_val
        })], ignore_index=True)

    data.to_csv(clean_data_path, index=False)

    return data


if __name__ == '__main__':
    clean()
