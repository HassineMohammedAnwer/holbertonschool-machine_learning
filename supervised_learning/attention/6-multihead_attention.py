#!/usr/bin/env python3
""" attention
"""
import tensorflow as tf
sdp_attention = __import__('5-sdp_attention').sdp_attention


class MultiHeadAttention(tf.keras.layers.Layer):
    """
    Perform multi-head attention.
    """

    def __init__(self, dm, h):
        """ dm (int): The dimensionality of the model.
            h (int): The number of heads. dm must be divisible by h.
        """
        super(MultiHeadAttention, self).__init__()
        if dm % h != 0:
            raise ValueError("dm must be divisible by h")

        self.h = h
        self.dm = dm
        self.depth = dm // h
        self.Wq = tf.keras.layers.Dense(units=dm)
        self.Wk = tf.keras.layers.Dense(units=dm)
        self.Wv = tf.keras.layers.Dense(units=dm)
        self.linear = tf.keras.layers.Dense(units=dm)

    def split_heads(self, x, batch_size):
        """split"""
        # Reshape to (batch_size, seq_len, h, depth)
        x = tf.reshape(x, (batch_size, -1, self.h, self.depth))
        # Transpose to (batch_size, h, seq_len, depth)
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, Q, K, V, mask):
        """multi-head"""
        batch_size = tf.shape(Q)[0]

        q = self.Wq(Q)
        k = self.Wk(K)
        v = self.Wv(V)

        q = self.split_heads(q, batch_size)
        k = self.split_heads(k, batch_size)
        v = self.split_heads(v, batch_size)

        scaled_attention, attention_weights = sdp_attention(q, k, v, mask)

        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])
        concat_attention = tf.reshape(scaled_attention,
                                      (batch_size, -1, self.dm))

        output = self.linear(concat_attention)
        return output, attention_weights
