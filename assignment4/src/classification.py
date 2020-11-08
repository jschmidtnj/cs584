#!/usr/bin/env python3
"""
classification (classification.py)
"""

from __future__ import annotations

import string
import yaml
from os.path import exists
from sys import argv
from typing import List, Optional, Any, Dict
from utils import file_path_relative
from variables import paragraph_key, clean_data_folder, models_folder, \
    cnn_folder, text_vectorization_folder, cnn_file_name, class_key
from loguru import logger
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa

from tensorflow.keras.layers.experimental.preprocessing import TextVectorization

# maximum number of words in vocabulary
vocab_size = 10000

# read dataset in batches of
batch_size = 10

# number of epochs to run
epochs = 10

# window size in cnn
window_size: int = 10



def create_text_vectorization_model(text_vectorization_filepath: str,
                                    dataset_all_tokens: tf.data.Dataset) -> tf.keras.models.Sequential:
    """
    create text vectorization model
    this vectorizer converts an array of strings to an array of integers
    """
    if exists(text_vectorization_filepath):
        logger.info('found text vectorization model')
        return tf.keras.models.load_model(text_vectorization_filepath, compile=False)

    vectorize_layer = TextVectorization(
        max_tokens=vocab_size,
        output_mode='int'
    )
    logger.success('created text vectorization layer')
    # batch the dataset to make it easier to store
    # in memory
    vectorize_layer.adapt(dataset_all_tokens.batch(batch_size))
    logger.success('adapted vectorization to training dataset')

    text_vectorization_model = tf.keras.models.Sequential([
        tf.keras.Input(shape=(1,), dtype=tf.string),
        vectorize_layer
    ])
    # simple text vectorization test
    logger.info(text_vectorization_model.predict(["this is a test"]))
    text_vectorization_model.save(text_vectorization_filepath)
    return text_vectorization_model


def build_cnn_model(size_data: int, current_batch_size=batch_size) -> tf.keras.models.Sequential:
    """
    build main cnn model

    batch_size is a parameter because it changes in testing
    """

    # TODO - build the model correctly
    model = tf.keras.models.Sequential([
        tf.keras.layers.Embedding(vocab_size, 64,
                                  batch_input_shape=[current_batch_size, size_data]),
        tf.keras.layers.LSTM(64, input_shape=(current_batch_size, size_data, 64), return_sequences=True),
        tf.keras.layers.Dense(2),
    ])
    logger.success('created tf model')
    return model


def get_tokens(text: str) -> List[str]:
    """
    remove punctuation, return list of words
    """
    text = text.translate(str.maketrans('', '', string.punctuation))
    return text.split(' ')


def flatten_input(data: List[List[Any]]) -> List[Any]:
    """
    flatten the given input (to 1xn)
    """
    return np.hstack(data).tolist()


def pad_zeros(data: List[int], num_elem: int) -> List[int]:
    """
    pads array so output is num_elem is length
    """
    if len(data) > num_elem:
        return data[:-num_elem]
    data.extend([0] * (num_elem - len(data)))
    return data


def cnn_train(name: str, clean_data: pd.DataFrame) -> tf.keras.models.Sequential:
    """
    cnn training
    creates the tensorflow cnn model for word prediction
    """
    logger.info(f'run cnn training for {name}')

    all_paragraphs: List[List[str]] = clean_data[paragraph_key]
    all_sentences: List[str] = flatten_input(all_paragraphs)
    all_tokens: List[str] = flatten_input([get_tokens(sentence) for sentence in all_sentences])
    dataset_all_tokens = tf.data.Dataset.from_tensor_slices(all_tokens)
    logger.success('created all tokens text dataset')

    # get text vectorization model
    text_vectorization_filepath = file_path_relative(f'{text_vectorization_folder}/{name}')

    text_vectorization_model = create_text_vectorization_model(
        text_vectorization_filepath, dataset_all_tokens)

    logger.info('get vectorized tokens')
    vectorized_paragraphs_file = file_path_relative(
        f'{clean_data_folder}/documents_vectorized.yml')
    vectorized_paragraphs: Optional[List[List[int]]] = None
    if exists(vectorized_paragraphs_file):
        logger.info('found vectorized paragraphs file')
        with open(vectorized_paragraphs_file, 'r') as yaml_file:
            vectorized_paragraphs = yaml.load(yaml_file, Loader=yaml.FullLoader)
    else:
        vectorized_paragraphs = [flatten_input(text_vectorization_model.predict(
            get_tokens(' '.join(paragraph)))) for paragraph in all_paragraphs]
        with open(vectorized_paragraphs_file, 'w') as yaml_file:
            yaml.dump(vectorized_paragraphs, yaml_file)

    # labels: List[int] = np.vstack(clean_data[class_key].to_numpy())
    labels: List[int] = clean_data[class_key].to_numpy()
    print(labels.shape)

    # create dataset
    length_vectorized_list = len(max(vectorized_paragraphs, key=len))
    vectorized_tokens_rectangular = [pad_zeros(paragraph, length_vectorized_list) for paragraph in vectorized_paragraphs]
    complete_dataset = tf.data.Dataset.from_tensor_slices((vectorized_tokens_rectangular, labels))
    logger.info('created complete dataset')

    # buffer size is used to shuffle the dataset
    buffer_size = 10000
    training_dataset = complete_dataset.shuffle(buffer_size).batch(
        batch_size, drop_remainder=True)
    logger.info('batched dataset')

    # print some samples
    logger.success('training data sample:')
    for input_example, target_example in training_dataset.take(1):
        logger.info(f"\ninput: {input_example}\ntarget: {target_example}")

    logger.info(f'training dataset shape: {training_dataset}')

    model = build_cnn_model(length_vectorized_list)

    model.compile(loss=tf.keras.losses.CategoricalCrossentropy(from_logits=False),
                    optimizer=tf.keras.optimizers.Adam(1e-4),
                    metrics=['accuracy'])
    logger.success('model compiled')

    cnn_filepath = file_path_relative(
        f'{cnn_folder}/{name}/{cnn_file_name}')

    # save checkpoints to disk
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=cnn_filepath,
        save_weights_only=True)

    # create visualizations
    _history = model.fit(training_dataset, epochs=epochs,
                        callbacks=[checkpoint_callback])
    model.summary()
    return text_vectorization_model


def cnn_predict_next(name: str,
                     clean_input_data: pd.DataFrame,
                     model: tf.keras.models.Sequential,
                     text_vectorization_model: tf.keras.models.Sequential,
                     num_lines_predict: Optional[int] = None,
                     num_predict: int = 1) -> None:
    """
    predict next word(s) with given input
    """

    logger.success(f'running predictions for {name}')

    predict_sentences: List[List[str]] = clean_input_data[paragraph_key]
    if num_lines_predict is not None:
        predict_sentences = predict_sentences[:num_lines_predict]

    # vectorize testing data
    vectorize_layer: TextVectorization = text_vectorization_model.layers[0]
    vocabulary = vectorize_layer.get_vocabulary()
    # logger.info(f'vocabulary: {vocabulary}')

    # reset model, get ready for predict
    model.reset_states()

    logger.success('[[<words>]] = predicted words:')

    sum_probability_log: float = 0.
    count_all_predict: int = 0

    # iterate over all input sentences
    for i, sentence in enumerate(predict_sentences):
        full_sentence = sentence.copy()
        for _ in range(num_predict):
            vectorized_sentence = flatten_input(text_vectorization_model.predict(
                full_sentence[-window_size:], batch_size=batch_size))
            input_eval = tf.expand_dims(vectorized_sentence, 0)
            predictions = model.predict(input_eval)
            # remove batch dimension, get probabilities of last word
            probabilities = tf.squeeze(predictions, 0)[-1]

            # get the index of the prediction based on the max probability
            predicted_index = np.argmax(probabilities)

            predicted_word = vocabulary[predicted_index]
            full_sentence.append(predicted_word)

            sum_probability_log += np.log(probabilities[predicted_index])
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
