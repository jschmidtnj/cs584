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
unseen_output: str = ''


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
    a sequence contains the first n - 1 words,
    and the next_count dictionary is a count of each of
    the next words.

    ex. n-grams: the stock is, the stock should
    sequence: the stock
    next_count dict: {
        is: 1, should: 1
    }

    total_count is the sum of all values in the next_count dict
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

    contains a dictionary of sequences to their corresponding
    objects (self.model). contains count of counts, total number
    of n-grams, and an inverse map of self.model (n_1_gram_map).
    """

    def __init__(self, n_grams: int):
        # n-grams number
        self.n_grams = n_grams

        # model of n-grams
        self.model: Dict[str, NGramsSequence] = {}

        # uninitialized aggregates

        # number of sequences that occur n-times
        # i.e. N_1 is the number of n-grams that occur 1 time
        self.count_map: Optional[Dict[int, int]] = None
        # sum of all counts
        self.total_count: Optional[int] = None
        # map between first n - 1 grams, and the n-grams
        self.n_1_gram_map: Optional[Dict[str, List[str]]] = None

    def create_count_map(self) -> Dict[int, int]:
        """
        return map of number of sequences (n-grams) with
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

    @staticmethod
    def get_n_minus_1_grams(n_grams: str) -> str:
        """
        get n minus 1 grams
        if n-gram is "start a paragraph",
        output is "start a"
        """
        return n_grams.rsplit(' ')[0]

    def create_n_1_gram_map(self) -> Dict[str, List[str]]:
        """
        create a map between the first n - 1 grams, and all of the n-grams
        that have that same n - 1 grams to start
        """
        assert self.count_map is not None, 'count map is not initialized'
        # assert self.n_grams > 1, 'n-grams must be greater than 1 in order to create n_1 gram map'

        res: Dict[str, List[str]] = {}
        for sequence in self.model:
            sequence: str = cast(str, sequence)
            n_minus_1_grams = self.get_n_minus_1_grams(sequence)
            if n_minus_1_grams not in res:
                res[n_minus_1_grams] = []
            res[n_minus_1_grams].append(sequence)

        self.n_1_gram_map = res
        return res

    def generate_aggregates(self) -> None:
        """
        create all the necessary aggregates
        """
        self.create_count_map()
        self.create_total_count()
        self.create_n_1_gram_map()

    @staticmethod
    def _basic_probability(count: int, sequence_total_count: int) -> float:
        """
        compute the probability for basic n-grams
        """
        return float(count) / sequence_total_count

    def _good_turing_new_c(self, count: int) -> float:
        """
        get good turing new count
        """
        next_count_index = count + 1
        next_count: Optional[float] = None
        if next_count_index not in self.count_map:
            # this happens when N_{c+1} is 0
            # this can make the total probability not equal to 1
            next_count = 0.
        else:
            next_count = float(self.count_map[next_count_index])

        new_count: Optional[float] = None
        new_count = (count + 1) * next_count / self.count_map[count]
        return new_count

    def _good_turing_smoothing_probability(self, count: int, sequence: str,
                                           sequence_total_count: int) -> float:
        """
        good turing smoothing implementation
        """
        assert self.count_map is not None and \
            self.total_count is not None, 'count map or total count not initialized'

        if sequence == unseen_output:
            # see slide 67 in lecture 5, green
            return self.count_map[1] / self.total_count

        new_count = self._good_turing_new_c(count)
        probability = new_count / sequence_total_count
        return probability

    def _kneser_ney_probability(self, count: int, sequence: str,
                                sequence_total_count: int) -> float:
        """
        kneser ney smoothing implementation
        """
        assert self.count_map is not None and \
            self.n_1_gram_map is not None, 'count map or n minus 1 gram map not initialized'

        count_previous_and_current: Optional[int] = None
        if sequence == unseen_output or sequence not in self.n_1_gram_map:
            # did not see given sequence, default count to 1
            count_previous_and_current = 1
        else:
            count_word = len(self.n_1_gram_map[sequence])
            count_previous_and_current = sequence_total_count + count_word
        d = count - self._good_turing_new_c(count)
        # first term is the term on the left of the equation
        first_term = max([count_previous_and_current - d, 0]
                         ) / float(sequence_total_count)

        if sequence == unseen_output:
            # if sequence is not seen, use frequency of unknown
            # lmbda = d / count * freq(unknown)
            sequence = unknown_token
        different_final_word_types: int = 0
        if sequence in self.model:
            current_sequence_data: NGramsSequence = self.model[sequence]
            different_final_word_types = len(current_sequence_data.next_count)
        # lambda is part of the second term
        lmbda = d / float(sequence_total_count) * different_final_word_types

        different_preceding_final_word_types: int = 0
        if sequence in self.n_1_gram_map:
            different_preceding_final_word_types = len(
                self.n_1_gram_map[sequence])

        num_n_grams = len(self.model)
        if num_n_grams == 0:
            return 0.

        # p_cont is the second part of the second term
        p_cont = float(different_preceding_final_word_types) / num_n_grams

        # return probability of the current sequence
        return first_term + lmbda * p_cont

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
            # get n_gram data from model
            current_sequence_data: NGramsSequence = self.model[sequence_input]
            if self.count_map is None:
                self.generate_aggregates()
            sequence_total_count = current_sequence_data.total_count
            current_counts = list(
                current_sequence_data.next_count.items()).copy()

        # iterate over all n_gram options for the given input
        for i, elem in enumerate(current_counts):
            sequence, count = elem
            probability: Optional[float] = None

            # switch statement for smoothing type
            if smoothing_type == SmoothingType.basic:
                probability = self._basic_probability(
                    count, sequence_total_count)
            elif smoothing_type == SmoothingType.good_turing:
                probability = self._good_turing_smoothing_probability(
                    count, sequence, sequence_total_count)
            elif smoothing_type == SmoothingType.kneser_ney:
                probability = self._kneser_ney_probability(
                    count, sequence, sequence_total_count)
            else:
                raise RuntimeError(
                    f'invalid smoothing type {smoothing_type.name} provided')

            current_counts[i] = sequence, probability

        # return sorted sequences based on probability (high to low)
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
        n_grams: int = data['n_grams']
        complete_model = cls(n_grams)

        model: Dict[str, NGramsSequence] = {}
        for sequence, sequence_obj in data['model'].items():
            model[sequence] = NGramsSequence.from_json(sequence, sequence_obj)
        complete_model.model = model

        complete_model.count_map = cls.dict_str_to_int(data['count_map'])
        complete_model.n_1_gram_map = data['n_1_gram_map']
        complete_model.total_count = data['total_count']

        return complete_model


def n_grams_train(name: str, file_name: Optional[str] = None,
                  clean_data: Optional[pd.DataFrame] = None,
                  n_grams: int = default_n_grams,
                  fill_in_blank: bool = False) -> NGramsModel:
    """
    n-grams training
    get a dictionary of grams to a dictionary of subsequent words and their counts
    """
    if file_name is None and clean_data is None:
        raise ValueError('no file name or tokens provided')

    # get training data
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

    if fill_in_blank and n_grams > 1:
        n_grams -= 1

    # create n-gram model
    n_grams_res = NGramsModel(n_grams)

    # train model with sliding window
    for sentence in tokens:
        for i in range(len(sentence) - n_grams):
            sequence = ' '.join(sentence[i: i + n_grams])
            if sequence not in n_grams_res.model:
                n_grams_res.model[sequence] = NGramsSequence(sequence)

            next_word = sentence[i + n_grams]
            n_grams_res.model[sequence].add_grams(next_word)

    # create aggregate objects
    n_grams_res.generate_aggregates()

    # save to disk
    json_file_path = file_path_relative(f'{models_folder}/{name}.json')
    with open(json_file_path, 'w', encoding='utf-8') as file:
        json.dump(n_grams_res, file, ensure_ascii=False,
                  indent=4, sort_keys=True)

    return n_grams_res


def n_grams_predict_next(name: str,
                         file_name: Optional[str] = None,
                         model: Dict[str, NGramsSequence] = None,
                         clean_input_file: Optional[str] = None,
                         clean_input_data: Optional[pd.DataFrame] = None,
                         num_lines_predict: Optional[int] = None,
                         n_grams: int = default_n_grams,
                         num_predict: int = 1,
                         smoothing: SmoothingType = SmoothingType.basic) -> None:
    """
    predict the next word(s) in the set
    """

    logger.success(f'predicting with {smoothing.name} for {name}')

    if file_name is None and model is None:
        raise ValueError('no file name or model provided')

    if clean_input_file is None and clean_input_data is None:
        raise ValueError('no input file name or data provided')

    # create n-gram model if not provided
    if model is None:
        json_file_path = file_path_relative(f'{models_folder}/{file_name}')
        logger.info(f'reading data from {json_file_path}')
        with open(json_file_path, 'r') as file:
            model = NGramsModel.from_json(json.load(file))

    # get testing data
    if clean_input_data is None:
        file_path = file_path_relative(
            f'{clean_data_folder}/{clean_input_file}')
        logger.info(f'reading data from {file_path}')
        clean_input_data = pd.read_csv(file_path, converters={
            sentences_key: literal_eval})

    predict_sentences: List[List[str]] = clean_input_data[sentences_key]
    if num_lines_predict is not None:
        predict_sentences = predict_sentences[:num_lines_predict]

    check_probability_smoothing: List[SmoothingType] = [SmoothingType.basic]

    logger.success('[[<words>]] = predicted words:')

    sum_probability_log: float = 0.
    count_all_predict: int = 0

    # iterate over testing data
    for i, sentence in enumerate(predict_sentences):
        full_sentence = sentence.copy()
        for _ in range(num_predict):
            last_words = full_sentence[-n_grams:]
            sequence = ' '.join(last_words)

            probabilities = model.get_probabilities(
                sequence, smoothing)
            sum_probability = sum(elem[1] for elem in probabilities)
            # logger.info(f'probabilities: sum: {sum_probability}, all: {probabilities}')
            if smoothing in check_probability_smoothing:
                # for not-unseen outputs, check to
                # make sure sum is approximately 1
                assert np.isclose(
                    sum_probability, 1), f'probability of {sum_probability} is not close to 1'

            current_output, prob = probabilities[0]
            full_sentence.append(current_output)
            # if not unseen, add to perplexity calculation
            if current_output != unseen_output:
                sum_probability_log += np.log(prob)
                count_all_predict += 1

        logger.info(
            f"{i + 1}. {' '.join(sentence)} [[{' '.join(full_sentence[len(sentence):])}]]")

    if count_all_predict == 0:
        logger.info('no predictions, no perplexity')
    else:
        total_loss = -1 * sum_probability_log
        perplexity: float = np.exp(total_loss / count_all_predict)
        logger.info(f"perplexity: {perplexity}")


if __name__ == '__main__':
    if len(argv) < 2:
        raise ValueError('no n-grams training data provided')
    n_grams_train(basename(splitext(argv[1])[0]), argv[1])
