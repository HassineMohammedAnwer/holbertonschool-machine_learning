#!/usr/bin/env python3
"""decode for machine translation"""
import tensorflow as tf
SelfAttention = __import__('1-self_attention').SelfAttention


class RNNDecoder(tf.keras.layers.Layer):
    """luiglig"""
    def __init__(self, vocab, embedding, units, batch):
        """constructor"""
        super(RNNDecoder, self).__init__()
        self.embedding = tf.keras.layers.Embedding(input_dim=vocab,
                                                   output_dim=embedding)
        self.gru = tf.keras.layers.GRU(units=units,
                                       recurrent_initializer='glorot_uniform',
                                       return_sequences=True,
                                       return_state=True)
        self.F = tf.keras.layers.Dense(vocab)
        self.attention = SelfAttention(units)

    def call(self, x, s_prev, hidden_states):
        """the decoder
        """
        #context vector
        context, attention_weights = self.attention(s_prev,
                                                hidden_states)

        # Embed previous input in target sequence
        x = self.embedding(x)

        # Concatenate context vector and embedded input
        comb = tf.concat([tf.expand_dims(context, 1), x], axis=-1)

        output, s = self.gru(comb, initial_state=s_prev)

        y = self.F(output)

        return y, s
