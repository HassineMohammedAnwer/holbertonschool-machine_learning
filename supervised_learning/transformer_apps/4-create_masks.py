#!/usr/bin/env python3
"""4. Create Masks"""
import tensorflow as tf


def create_masks(inputs, target):
    """
    creates all masks for training/validation:
    inputs is a tf.Tensor of shape (batch_size, seq_len_in)
    __that contains the input sentence
    target is a tf.Tensor of shape (batch_size, seq_len_out)
    __that contains the target sentence
    This function should only use tensorflow operations in
    __order to properly function in the training step
    Returns: encoder_mask, combined_mask, decoder_mask
    encoder_mask is the tf.Tensor padding mask of shape
    __(batch_size, 1, 1, seq_len_in) to be applied in the encoder
    combined_mask is the tf.Tensor of shape
    ___batch_size, 1, seq_len_out, seq_len_out) used in the 1st
    __attention block in the decoder to pad and mask future tokens
    __in the input received by the decoder. It takes the maximum
    __between a lookaheadmask and the decoder target padding mask.
    decoder_mask is the tf.Tensor padding mask of shape
    __(batch_size, 1, 1, seq_len_in) used in the 2nd attention block in the decoder.
    """
    encoder_m = tf.cast(tf.math.equal(inputs, 0), tf.float32)
    encoder_m = encoder_m[:, tf.newaxis, tf.newaxis, :]
    decoder_m = tf.cast(tf.math.equal(inputs, 0), tf.float32)
    decoder_m = decoder_m[:, tf.newaxis, tf.newaxis, :]
    size = tf.shape(target)[1]
    look_ahead_mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
    dec_target_padding_mask = tf.cast(tf.math.equal(target, 0), tf.float32)
    dec_target_padding_mask = dec_target_padding_mask[:, tf.newaxis, tf.newaxis, :]
    combined_m = tf.maximum(dec_target_padding_mask, look_ahead_mask)
    return encoder_m, combined_m, decoder_m
