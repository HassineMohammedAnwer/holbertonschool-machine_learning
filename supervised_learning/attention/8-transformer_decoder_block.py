#!/usr/bin/env python3
"""
7. Transformer Decoder Block"""

import tensorflow as tf
MultiHeadAttention = __import__('6-multihead_attention').MultiHeadAttention


class DecoderBlock(tf.keras.layers.Layer):
    """
    decoder"""
    def __init__(self, dm, h, hidden, drop_rate=0.1):
        """
        dm (int): The dimensionality of the model.
        h (int): The number of heads.
        hidden (int): The number of hidden units in the fully connected
                      layer.
        drop_rate (float): The dropout rate.
        """
        super(DecoderBlock, self).__init__()
        self.mha1 = MultiHeadAttention(dm, h)
        self.mha2 = MultiHeadAttention(dm, h)
        self.dense_hidden = tf.keras.layers.Dense(units=hidden,
                                                  activation='relu')
        self.dense_output = tf.keras.layers.Dense(units=dm)
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = tf.keras.layers.Dropout(rate=drop_rate)
        self.dropout2 = tf.keras.layers.Dropout(rate=drop_rate)
        self.dropout3 = tf.keras.layers.Dropout(rate=drop_rate)

    def call(self, x, encoder_output, training, look_ahead_mask, padding_mask):
        """
        forward pass
        Args:
            x (tf.Tensor): Input tensor to the decoder block
                           shape (batch, target_seq_len, dm).
            encoder_output (tf.Tensor): Output tensor from the encoder
                                        shape (batch, input_seq_len, dm).
            training (bool): Boolean to determine if the model is training.
            look_ahead_mask: Mask applied to the first MHA layer.
            padding_mask: Mask applied to the second MHA layer.

        Returns:
            tf.Tensor: Output tensor of shape (batch, target_seq_len, dm).
        """
        attn1, attn_weights_block1 = self.mha1(x, x, x, look_ahead_mask)
        attn1 = self.dropout1(attn1, training=training)
        out1 = self.layernorm1(x + attn1)
        attn2, attn_weights_block2 = self.mha2(out1, encoder_output,
                                               encoder_output, padding_mask)
        attn2 = self.dropout2(attn2, training=training)
        out2 = self.layernorm2(out1 + attn2)
        ffn_hidden_output = self.dense_hidden(out2)
        ffn_output = self.dense_output(ffn_hidden_output)
        ffn_output = self.dropout3(ffn_output, training=training)
        out3 = self.layernorm3(out2 + ffn_output)
        return out3
