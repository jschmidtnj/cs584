#!/usr/bin/env python3
"""
encoder file

encoder class
"""

import tensorflow as tf


class Encoder(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dimension, encoding_units, batch_size, gru: bool = True):
        """
        encoder object
        """
        super().__init__()
        self.batch_size = batch_size
        self.encoding_units = encoding_units
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dimension)
        if gru:
            self.layer = tf.keras.layers.GRU(self.encoding_units,
                                             return_sequences=True,
                                             return_state=True,
                                             recurrent_initializer='glorot_uniform')
        else:
            self.layer = tf.keras.layers.LSTM(self.encoding_units,
                                              return_sequences=True,
                                              return_state=True)

    def call(self, x, hidden):
        """
        run embedding and gru layers
        """
        x = self.embedding(x)
        output, state = self.layer(x, initial_state=hidden)
        return output, state

    def initialize_hidden_state(self):
        """
        create initial hidden state
        """
        return tf.zeros((self.batch_size, self.encoding_units))
