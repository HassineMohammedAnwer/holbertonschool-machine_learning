#!/usr/bin/env python3
"""
8. Transformer Encoder"""

import tensorflow as tf
positional_encoding = __import__('4-positional_encoding').positional_encoding
EncoderBlock = __import__('7-transformer_encoder_block').EncoderBlock


class Encoder(tf.keras.layers.Layer):
    """
    encoder
    """
    def __init__(self, N, dm, h, hidden, input_vocab,
                 max_seq_len, drop_rate=0.1):
        """
        N (int): The number of blocks in the encoder.
        dm (int): The dimensionality of the model.
        h (int): The number of heads.
        hidden (int): The number of hidden units in the fully connected
                      layer within EncoderBlock.
        input_vocab (int): The size of the input vocabulary.
        max_seq_len (int): The maximum sequence length possible.
        drop_rate (float): The dropout rate.
        """
        super(Encoder, self).__init__()
        self.N = N
        self.dm = dm
        self.embedding = tf.keras.layers.Embedding(input_vocab, dm)
        self.positional_encoding = positional_encoding(max_seq_len, dm)
        self.blocks = [EncoderBlock(dm, h, hidden, drop_rate)
                       for _ in range(N)]
        self.dropout = tf.keras.layers.Dropout(drop_rate)

    def call(self, x, training, mask):
        """
        forward pass
        Args:
            x (tf.Tensor): Input tensor of shape (batch, input_seq_len)
                           containing input token indices.
            training (bool): Boolean to determine if the model is training.
            mask: Mask to be applied for multi head attention within blocks.

        Returns:
            tf.Tensor: Output tensor of shape (batch, input_seq_len, dm).
        """
        seq_len = tf.shape(x)[1]
        x = self.embedding(x)
        x *= tf.math.sqrt(tf.cast(self.dm, tf.float32))
        x += self.positional_encoding[:seq_len]
        x = self.dropout(x, training=training)
        for i in range(self.blocks):
            (x, training, mask)
        return x
