#!/usr/bin/env python3
"""
ngrams (ngrams.py)
"""

from __future__ import annotations

from sys import argv
from os.path import basename, splitext
from typing import List, Optional, Dict, Tuple, Any, cast
from utils import file_path_relative
from variables import clean_data_folder, models_folder, sentences_key, unknown_token
from loguru import logger
from ast import literal_eval
import pandas as pd
import numpy as np
import json_fix  # pylint: disable=unused-import
import json
from enum import Enum


# default n gram size
default_n_grams: int = 2

# output if the input sequence is not found in the training data
unseen_output: str = '<unseen>'


class SmoothingType(Enum):
    """
    Enum to store smoothing types for ngrams
    """
    basic = "basic"
    good_turing = "good_turing"
    kneser_ney = "kneser_ney"


class NGramsSequence:
    """
    n-grams sequence object
    """

    def __init__(self, sequence: str):
        self.sequence = sequence
        self.next_count: Dict[str, int] = {}
        self.total_count: int = 0

    def add_grams(self, grams: str):
        """
        add gram to sequence
        """
        if grams not in self.next_count:
            self.next_count[grams] = 0
        self.next_count[grams] += 1
        self.total_count += 1

    def to_json(self) -> Dict:
        """
        return json data
        """
        return self.__dict__

    def __str__(self):
        return json.dumps(self.to_json())

    def __repr__(self):
        return self.__str__()

    @classmethod
    def from_json(cls, sequence_name: str, data: Dict[str, Any]):
        """
        return NGramsSequence object from dict
        """
        sequence = cls(sequence_name)
        sequence.total_count = data['total_count']
        sequence.next_count = data['next_count']
        return sequence


class NGramsModel:
    """
    model class for encapsulating n-grams data
    """

    def __init__(self):
        # number of sequences that occur n-times
        # i.e. N_1 is the number of n-grams that occur 1 time
        self.count_map: Optional[Dict[int, int]] = None
        # sum of all counts
        self.total_count: Optional[int] = None
        # model of n-grams
        self.model: Dict[str, NGramsSequence] = {}

    def create_count_map(self) -> Dict[int, int]:
        """
        return map of number of sequences with
        each count
        """
        res: Dict[int, int] = {}
        for sequence_data in self.model.values():
            sequence_data: NGramsSequence = cast(NGramsSequence, sequence_data)
            for count in sequence_data.next_count.values():
                count: int = cast(int, count)
                if count not in res:
                    res[count] = 0
                res[count] += 1
        self.count_map = res
        logger.success('created count map')
        return res

    def create_total_count(self) -> int:
        """
        get count of all n-grams
        """
        assert self.count_map is not None, 'count map is not initialized'

        res = sum(self.count_map.values())
        self.total_count = res
        return res

    def generate_aggregates(self) -> None:
        """
        create all the necessary aggregates
        """
        self.create_count_map()
        self.create_total_count()

    def _good_turing_smoothing_probability(self, count: int, sequence: str,
                                           sequence_total_count: int) -> float:
        """
        good turing smoothing implementation
        """

        assert self.count_map is not None and \
            self.total_count is not None, 'count map or total count not initialized'

        if sequence == unseen_output:
            # TODO - ask if this is correct - see slide 67 in lecture 5, green
            # zero frequency, use N1
            return self.count_map[1] / self.total_count

        next_count_index = count + 1
        next_count: Optional[float] = None
        if next_count_index not in self.count_map:
            # this should not ever happen
            next_count = 0.
        else:
            next_count = float(self.count_map[next_count_index])

        new_count: Optional[float] = None
        new_count = (count + 1) * next_count / self.count_map[count]
        prob = new_count / sequence_total_count
        # print(prob)
        return prob

    def _kneser_ney_smoothing_probability(self, count: int) -> float:
        """
        run kneser ney smoothing algorithm
        """
        return 0.

    def get_probabilities(self, sequence_input: str,
                          smoothing_type: SmoothingType) -> List[Tuple[str, float]]:
        """
        get all probabilities of next elem given sequence
        """
        sequence_total_count: Optional[int] = None
        current_counts: Optional[List[Tuple[str, int]]] = None
        if sequence_input not in self.model:
            # handle unknown input
            sequence_total_count = 1
            current_counts = [(unseen_output, sequence_total_count)]
        else:
            current_sequence_data: NGramsSequence = self.model[sequence_input]
            if self.count_map is None:
                self.generate_aggregates()
            sequence_total_count = current_sequence_data.total_count
            current_counts = list(
                current_sequence_data.next_count.items()).copy()

        for i, elem in enumerate(current_counts):
            sequence, count = elem
            probability: Optional[float] = None

            if smoothing_type == SmoothingType.basic:
                probability = float(count) / sequence_total_count
            elif smoothing_type == SmoothingType.good_turing:
                probability = self._good_turing_smoothing_probability(
                    count, sequence, sequence_total_count)
            elif smoothing_type == SmoothingType.kneser_ney:
                probability = self._kneser_ney_smoothing_probability(count)

            current_counts[i] = sequence, probability

        sorted_counts = sorted(current_counts,
                               key=lambda elem: elem[1], reverse=True)

        return sorted_counts

    def to_json(self) -> Dict:
        """
        return json data
        """
        return self.__dict__

    def __str__(self):
        return json.dumps(self.to_json())

    def __repr__(self):
        return self.__str__()

    @staticmethod
    def dict_str_to_int(obj: Dict[str, Any]) -> Dict[int, Any]:
        """
        dict with string keys to int
        """
        res: Dict[str, Any] = {}
        for key, val in obj.items():
            res[int(key)] = val
        return res

    @classmethod
    def from_json(cls, data: Dict[str, Any]) -> NGramsModel:
        """
        return NGramsSequence object from dict
        """
        complete_model = cls()

        model: Dict[str, NGramsSequence] = {}
        for sequence, sequence_obj in data['model'].items():
            model[sequence] = NGramsSequence.from_json(sequence, sequence_obj)
        complete_model.model = model

        complete_model.count_map = cls.dict_str_to_int(data['count_map'])

        return complete_model


def n_grams_train(name: str, file_name: Optional[str] = None,
                  clean_data: Optional[pd.DataFrame] = None,
                  n_grams: int = default_n_grams) -> NGramsModel:
    """
    n-grams training
    get a dictionary of grams to a dictionary of subsequent words and their counts
    """
    if file_name is None and clean_data is None:
        raise ValueError('no file name or tokens provided')

    if clean_data is None:
        file_path = file_path_relative(f'{clean_data_folder}/{file_name}')
        logger.info(f'reading data from {file_path}')
        clean_data = pd.read_csv(file_path, converters={
            sentences_key: literal_eval})

    tokens: List[List[str]] = clean_data[sentences_key]
    average_sentence_len = np.average([len(sentence) for sentence in tokens])
    if average_sentence_len < n_grams:
        raise ValueError(
            f'n-grams of {n_grams} is greater than average sentence ' +
            f'length of {average_sentence_len} in training data')

    n_grams_res = NGramsModel()

    for sentence in tokens:
        for i in range(len(sentence) - n_grams):
            sequence = ' '.join(sentence[i: i + n_grams])
            if sequence not in n_grams_res.model:
                n_grams_res.model[sequence] = NGramsSequence(sequence)

            next_word = sentence[i + n_grams]
            n_grams_res.model[sequence].add_grams(next_word)

    n_grams_res.generate_aggregates()

    json_file_path = file_path_relative(f'{models_folder}/{name}.json')
    with open(json_file_path, 'w', encoding='utf-8') as file:
        json.dump(n_grams_res, file, ensure_ascii=False,
                  indent=4, sort_keys=True)

    return n_grams_res


def n_grams_predict_next(file_name: Optional[str] = None,
                         model: Dict[str, NGramsSequence] = None,
                         clean_input_file: Optional[str] = None,
                         clean_input_data: Optional[pd.DataFrame] = None,
                         num_lines_predict: int = 30,
                         n_grams: int = default_n_grams,
                         num_predict: int = 1,
                         smoothing: SmoothingType = SmoothingType.basic) -> None:
    """
    predict the next word in the set
    """

    logger.success(f'predicting with {smoothing.name}')

    if file_name is None and model is None:
        raise ValueError('no file name or model provided')

    if clean_input_file is None and clean_input_data is None:
        raise ValueError('no input file name or data provided')

    if model is None:
        json_file_path = file_path_relative(f'{models_folder}/{file_name}')
        logger.info(f'reading data from {json_file_path}')
        with open(json_file_path, 'r') as file:
            model = NGramsModel.from_json(json.load(file))

    if clean_input_data is None:
        file_path = file_path_relative(
            f'{clean_data_folder}/{clean_input_file}')
        logger.info(f'reading data from {file_path}')
        clean_input_data = pd.read_csv(file_path, converters={
            sentences_key: literal_eval})

    predict_sentences: List[List[str]] = clean_input_data[sentences_key]

    for i, sentence in enumerate(predict_sentences[:num_lines_predict]):
        full_sentence = sentence.copy()
        for _ in range(num_predict):
            last_words = full_sentence[-n_grams:]
            sequence = ' '.join(last_words)

            probabilities = model.get_probabilities(
                sequence, smoothing)

            current_output, _prob = probabilities[0]
            if current_output != unseen_output:
                # for not-unseen outputs, check to
                # make sure sum is approximately 1
                sum_probability = sum(elem[1] for elem in probabilities)
                # print(probabilities)
                assert np.isclose(
                    sum_probability, 1), f'probability of {sum_probability} is not close to 1'

            current_output, _prob = probabilities[0]
            full_sentence.append(current_output)

        logger.info(f"{i + 1}. input: {' '.join(sentence)}, "
                    + f"predicted: {' '.join(full_sentence[len(sentence):])}")


if __name__ == '__main__':
    if len(argv) < 2:
        raise ValueError('no n-grams training data provided')
    n_grams_train(basename(splitext(argv[1])[0]), argv[1])
