#!/usr/bin/env python3
"""
rnn (rnn.py)
"""

from __future__ import annotations

from sys import argv
from typing import List, Optional, Tuple, Any
from utils import file_path_relative
from variables import sentences_key, clean_data_folder, checkpoints_folder
from loguru import logger
from ast import literal_eval
import pandas as pd
import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa

from tensorflow.keras.layers.experimental.preprocessing import TextVectorization

# maximum number of words in vocabulary
vocab_size = 10000

# read dataset in batches of
batch_size = 50

# number of epochs to run
epochs = 10

# window size in rnn
window_size: int = 20

checkpoint_filepath = file_path_relative(
    f'{checkpoints_folder}/cp.ckpt')

text_vectorization_filepath = file_path_relative(
    f'{checkpoints_folder}/vectorization')


def create_text_vectorization_model(dataset_all_tokens: tf.data.Dataset) -> tf.keras.models.Sequential:
    """
    create text vectorization model
    """
    vectorize_layer = TextVectorization(
        max_tokens=vocab_size,
        output_mode='int',
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
    logger.info(text_vectorization_model.predict(["test it help"]))
    text_vectorization_model.save(text_vectorization_filepath)
    return text_vectorization_model


def build_model() -> tf.keras.models.Sequential:
    """
    build main rnn model
    """
    # rnn params
    embedding_dim = 256
    rnn_units = 1024

    model = tf.keras.models.Sequential([
        tf.keras.layers.Embedding(vocab_size, embedding_dim,
                                  batch_input_shape=[batch_size, None]),
        tf.keras.layers.GRU(rnn_units,
                            return_sequences=True,
                            stateful=True,
                            recurrent_initializer='glorot_uniform'),
        tf.keras.layers.Dense(vocab_size)
    ])
    logger.success('created tf model')
    model.save()
    return model


def flatten_input(data: List[List[Any]]) -> List[Any]:
    """
    flatten the given input
    """
    return np.hstack(data).tolist()


def rnn_train(name: str, file_name: Optional[str] = None,
              clean_data: Optional[pd.DataFrame] = None) -> Tuple[tf.keras.models.Sequential, tf.keras.models.Sequential]:
    """
    rnn training
    creates the tensorflow rnn model for word prediction
    """
    logger.info(f'run rnn training for {name}')

    if file_name is None and clean_data is None:
        raise ValueError('no file name or tokens provided')

    if clean_data is None:
        file_path = file_path_relative(f'{clean_data_folder}/{file_name}')
        logger.info(f'reading data from {file_path}')
        clean_data = pd.read_csv(file_path, converters={
            sentences_key: literal_eval})

    tokens: List[List[str]] = clean_data[sentences_key]
    flattened_tokens: List[str] = flatten_input(tokens)
    dataset_all_tokens = tf.data.Dataset.from_tensor_slices(flattened_tokens)
    logger.success('created all tokens text dataset')

    text_vectorization_model = create_text_vectorization_model(
        dataset_all_tokens)
    vectorized_tokens: List[int] = flatten_input(text_vectorization_model.predict(
        flattened_tokens, batch_size=batch_size))

    training_data: List[Tuple[List[str], List[str]]] = []
    for i in range(len(vectorized_tokens) - window_size - 1):
        vectorized_sequence = vectorized_tokens[i: i + window_size]
        target = [vectorized_tokens[i + window_size]]
        target.extend([0] * (window_size - len(target)))
        training_data.append(tf.tuple((vectorized_sequence, target)))

    training_dataset = tf.data.Dataset.from_tensor_slices(training_data)

    def reshape_tuple(elem):
        """
        reshape the dataset to have tuples
        """
        return elem[0], elem[1]
    training_dataset = training_dataset.map(reshape_tuple)

    for input_example, target_example in training_dataset.take(3):
        logger.info(f"\ninput: {input_example}\ntarget: {target_example}")

    # buffer size is used to shuffle the dataset
    buffer_size = 10000
    training_dataset = training_dataset.shuffle(
        buffer_size).batch(batch_size, drop_remainder=True)
    logger.info(f'training dataset: {training_dataset}')

    model = build_model()

    def loss(targets, logits):
        """
        return loss for given iteration
        """
        return tfa.seq2seq.sequence_loss(logits, targets, tf.ones([batch_size, window_size]))

    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
    model.compile(optimizer=optimizer, loss=loss)
    logger.success('model compiled')

    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_filepath,
        save_weights_only=True)

    history = model.fit(training_dataset, epochs=epochs,
                        callbacks=[checkpoint_callback])
    model.summary()
    return model, text_vectorization_model


def rnn_predict_next(name: str,
                     model: tf.keras.models.Sequential = None,
                     text_vectorization_model: tf.keras.models.Sequential = None,
                     clean_input_file: Optional[str] = None,
                     clean_input_data: Optional[pd.DataFrame] = None,
                     num_lines_predict: Optional[int] = None,
                     num_predict: int = 1) -> None:
    """
    predict next word(s) with given input
    """

    logger.success(f'running predictions for {name}')

    if clean_input_file is None and clean_input_data is None:
        raise ValueError('no input file name or data provided')

    if model is None:
        model = build_model()
        model.load_weights(checkpoint_filepath)

    if text_vectorization_model is None:
        text_vectorization_model = tf.keras.models.load_model(
            text_vectorization_filepath)

    if clean_input_data is None:
        file_path = file_path_relative(
            f'{clean_data_folder}/{clean_input_file}')
        logger.info(f'reading data from {file_path}')
        clean_input_data = pd.read_csv(file_path, converters={
            sentences_key: literal_eval})

    predict_sentences: List[List[str]] = clean_input_data[sentences_key]
    if num_lines_predict is not None:
        predict_sentences = predict_sentences[:num_lines_predict]

    vectorize_layer: TextVectorization = text_vectorization_model.layer[0]
    vocabulary = vectorize_layer.get_vocabulary()

    for i, sentence in enumerate(predict_sentences):
        full_sentence = sentence.copy()
        logger.info(f"{i + 1}. input: {' '.join(sentence)}")
        for _ in range(num_predict):
            vectorized_sentence: List[int] = flatten_input(text_vectorization_model.predict(
                full_sentence[-window_size:], batch_size=batch_size))
            output = model.predict(vectorized_sentence)
            logger.info(output)
            word = vocabulary[output]
            full_sentence.append(output)

        logger.info(f"predicted: {' '.join(full_sentence[len(sentence):])}")



if __name__ == '__main__':
    if len(argv) < 2:
        raise ValueError('no n-grams training data provided')
