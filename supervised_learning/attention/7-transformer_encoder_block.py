#!/usr/bin/env python3
""" attention
"""
import tensorflow as tf
MultiHeadAttention = __import__('6-multihead_attention').MultiHeadAttention


class EncoderBlock(tf.keras.layers.Layer):
    """
    encoder"""
    def __init__(self, dm, h, hidden, drop_rate=0.1):
        """ dm (int): The dimensionality of the model.
            h (int): The number of heads.
            hidden (int): The number of hidden units in the fully connected
                          layer.
            drop_rate (float): The dropout rate.
        """
        super(EncoderBlock, self).__init__()

        self.mha = MultiHeadAttention(dm, h)

        self.dense_hidden = tf.keras.layers.Dense(units=hidden,
                                                  activation='relu')
        self.dense_output = tf.keras.layers.Dense(units=dm)

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = tf.keras.layers.Dropout(rate=drop_rate)
        self.dropout2 = tf.keras.layers.Dropout(rate=drop_rate)

    def call(self, x, training, mask=None):
        """
        forward pass
            x (tf.Tensor): Input tensor of shape (batch, input_seq_len, dm).
            training (bool): Boolean to determine if the model is training.
            mask: Mask to be applied for multi head attention.
        Returns:
            tf.Tensor: Output tensor of shape (batch, input_seq_len, dm).
        """
        attn_output, _ = self.mha(x, x, x, mask)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(x + attn_output)
        ffn_hidden_output = self.dense_hidden(out1)
        ffn_output = self.dense_output(ffn_hidden_output)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.layernorm2(out1 + ffn_output)
        return out2
