#!/usr/bin/env python3
"""
main file

entry point for running assignment 1
"""

from clean import clean_tokenize
from ngrams import n_grams_train, n_grams_predict_next, SmoothingType


def main() -> None:
    """
    main entry point
    """
    small_train_name: str = 'train.5k'
    small_train_data = clean_tokenize(f'{small_train_name}.txt')[0]
    train_small_model = n_grams_train(
        small_train_name, clean_data=small_train_data)
    validation_data = clean_tokenize('valid.txt')[0]
    n_grams_predict_next(model=train_small_model, clean_input_data=validation_data,
                         smoothing=SmoothingType.good_turing)


if __name__ == '__main__':
    main()
