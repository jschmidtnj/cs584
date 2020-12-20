#!/usr/bin/env python3
"""
decoder file

decoder class
"""

import tensorflow as tf


class BahdanauAttention(tf.keras.layers.Layer):
    def __init__(self, units):
        """
        attention layer from Bahdanau paper
        """
        super().__init__()
        self.w1 = tf.keras.layers.Dense(units)
        self.w2 = tf.keras.layers.Dense(units)
        self.vector = tf.keras.layers.Dense(1)

    def call(self, query, values):
        """
        get context and weights given query and values
        """
        query_with_time_axis = tf.expand_dims(query, 1)

        score = self.vector(tf.nn.tanh(
            self.w1(query_with_time_axis) + self.w2(values)))

        attention_weights = tf.nn.softmax(score, axis=1)

        context_vector = tf.reduce_sum(attention_weights * values, axis=1)

        return context_vector, attention_weights


class Decoder(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dimension, decoding_units, batch_size, gru: bool = True):
        """
        decoder for attention model
        """
        super().__init__()
        self.batch_size = batch_size
        self.decoding_units = decoding_units
        self.embedding = tf.keras.layers.Embedding(
            vocab_size, embedding_dimension)
        if gru:
            self.layer = tf.keras.layers.GRU(self.decoding_units,
                                             return_sequences=True,
                                             return_state=True,
                                             recurrent_initializer='glorot_uniform')
        else:
            self.layer = tf.keras.layers.LSTM(self.decoding_units,
                                              return_sequences=True,
                                              return_state=True)
        self.dense_layer = tf.keras.layers.Dense(vocab_size)
        self.attention = BahdanauAttention(self.decoding_units)

    def call(self, x, hidden, enc_output):
        """
        given vector, hidden, and encoding, return new vector, state, and weights
        """
        context_vector, attention_weights = self.attention(hidden, enc_output)

        x = self.embedding(x)

        x = tf.concat([tf.expand_dims(context_vector, 1), x], -1)

        output, state = self.layer(x)

        output = tf.reshape(output, (-1, output.shape[2]))

        x = self.dense_layer(output)

        return x, state, attention_weights
