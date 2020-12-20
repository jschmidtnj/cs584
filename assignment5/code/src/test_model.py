#!/usr/bin/env python3
"""
test file

run test on dataset
"""

import tensorflow as tf
import numpy as np
from loguru import logger
from data import preprocess_sentence, preprocess_without_tokens
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from typing import List
from nltk.translate.bleu_score import sentence_bleu
from variables import output_folder
from utils import file_path_relative


def _evaluate(sentence, max_length_targ, max_length_inp, inp_lang, targ_lang, units, encoder, decoder):
    """
    run evaluation with given input
    """
    attention_plot = np.zeros((max_length_targ, max_length_inp))

    sentence = preprocess_sentence(sentence)

    inputs = []
    for word in sentence.split(' '):
        if word not in inp_lang.word_index:
            logger.info(f'cannot find word {word} in trained words')
        else:
            inputs.append(inp_lang.word_index[word])
    inputs = tf.keras.preprocessing.sequence.pad_sequences([inputs],
                                                           maxlen=max_length_inp,
                                                           padding='post')
    inputs = tf.convert_to_tensor(inputs)

    output: List[str] = []

    hidden = [tf.zeros((1, units))]
    enc_out, enc_hidden = encoder(inputs, hidden)

    dec_hidden = enc_hidden
    dec_input = tf.expand_dims([targ_lang.word_index['<start>']], 0)

    for t in range(max_length_targ):
        predictions, dec_hidden, attention_weights = decoder(dec_input,
                                                             dec_hidden,
                                                             enc_out)

        # storing the attention weights to plot later on
        attention_weights = tf.reshape(attention_weights, (-1, ))
        attention_plot[t] = attention_weights.numpy()

        predicted_id = tf.argmax(predictions[0]).numpy()

        if targ_lang.index_word[predicted_id] == '<end>':
            return output, sentence, attention_plot

        output.append(targ_lang.index_word[predicted_id])

        # the predicted ID is fed back into the model
        dec_input = tf.expand_dims([predicted_id], 0)

    return output, sentence, attention_plot


def _plot_attention(attention, sentence, predicted_sentence, name: str):
    """
    plot the attention weights
    """
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(1, 1, 1)
    ax.matshow(attention, cmap='viridis')

    fontdict = {'fontsize': 14}

    ax.set_xticklabels([''] + sentence, fontdict=fontdict, rotation=90)
    ax.set_yticklabels([''] + predicted_sentence, fontdict=fontdict)

    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

    plt.show()
    file_path = file_path_relative(
        f'{output_folder}/attention_{name}.jpg')
    plt.savefig(file_path)


def run_tests(max_length_targ, max_length_inp, inp_lang, targ_lang, units, encoder, decoder, input_vals: List[str], target_vals: List[str], name: str, first_print: int = 5) -> float:
    """
    calculate the bleu score and output some input output data
    """
    scores: List[float] = []
    for i, line in enumerate(input_vals):
        output, sentence, attention_plot = _evaluate(line, max_length_targ, max_length_inp, inp_lang, targ_lang, units, encoder, decoder)
        if i < first_print:
            logger.info(f'Input: {sentence}')
            logger.info(f"Translated: {' '.join(output)}")
            attention_plot = attention_plot[:len(output), :len(sentence.split(' '))]
            _plot_attention(attention_plot, sentence.split(' '), output, f'{name}_{i}')

        target_val = preprocess_without_tokens(target_vals[i])
        current_score = sentence_bleu([output], target_val)
        scores.append(current_score)
    total_score = np.average(scores)
    logger.info(f'bleu score: {total_score}')

    return total_score


if __name__ == '__main__':
    raise RuntimeError('cannot run lstm on its own')
