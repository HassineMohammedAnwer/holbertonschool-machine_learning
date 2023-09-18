#!/usr/bin/env python3
"""Encode for machine translation"""
import tensorflow as tf


class RNNEncoder(tf.keras.layers.Layer):
    """param: --> self , vocab, embedding, units, batch"""

    def __init__(self, vocab, embedding, units, batch):
        """ml;m
        """
        super(RNNEncoder, self).__init__()
        self.batch = batch
        self.units = units
        self.embedding = tf.keras.layers.Embedding(vocab, embedding)
        self.gru = tf.keras.layers.GRU(units,
                                       recurrent_initializer='glorot_uniform',
                                       return_sequences=True,
                                       return_state=True)

    def initialize_hidden_state(self):
        """returns initialized hidden states
        """
        return tf.keras.initializers.Zeros()(shape=(self.batch, self.units))

    def call(self, x, initial):
        """returns outputs of the encoder and its last hidden state
         """
        outputs, hidden = self.gru(self.embedding(x), initial_state=initial)
        return outputs, hidden
