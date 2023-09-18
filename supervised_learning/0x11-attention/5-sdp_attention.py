#!/usr/bin/env python3

import tensorflow as tf


def sdp_attention(Q, K, V, mask=None):
    """the scaled dot product attention."""
    product = tf.matmul(Q, K, transpose_b=True)

    # Scale
    Qk = tf.matmul(Q, K, transpose_b=True)
    dk = tf.cast(tf.shape(K)[-1], tf.float32)
    sc = Qk / tf.math.sqrt(dk)

    # Apply the mask if provided
    if mask is not None:
        sc += (mask * -1e9)

    # Compute the attention weights
    w_attention = tf.nn.softmax(sc, axis=-1)

    # Calculate the weighted sum of V
    output = tf.matmul(w_attention, V)

    return output, w_attention
