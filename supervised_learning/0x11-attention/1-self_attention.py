#!/usr/bin/env python3
"""calculate the attention for machine translation"""
import tensorflow as tf


class SelfAttention(tf.keras.layers.Layer):
    """luiglig"""
    def __init__(self, units):
        """constructor"""
        super(SelfAttention, self).__init__()
        self.W = tf.keras.layers.Dense(units)
        self.U = tf.keras.layers.Dense(units)
        self.V = tf.keras.layers.Dense(1)

    def call(self, s_prev, hidden_states):
        """compute the alignment scores, weights, and context"""
        s_prev_dims = tf.expand_dims(s_prev, 1)
        #Alignment scores. Pass them through tanh function
        scores = self.V(tf.nn.tanh(self.W(s_prev_dims) + self.U(hidden_states)))
        #Compute the weights
        w_attention = tf.nn.softmax(scores, axis=1)
        #Compute the context vector
        context = w_attention * hidden_states
        context = tf.reduce_sum(context, axis=1)

        return context, w_attention
