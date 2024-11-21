#!/usr/bin/env python3
"""3. Extract Word2Vec"""
import tensorflow as tf


def gensim_to_keras(model):
    """ converts a gensim word2vec model to a keras Embedding layer:
    model is a trained gensim word2vec models
    Returns: the trainable keras Embedding
    Note : the weights can / will be further updated in Keras."""
    keyed_vectors = model.wv
    weights = keyed_vectors.vectors    
    layer = tf.keras.layers.Embedding(
        input_dim=weights.shape[0],
        output_dim=weights.shape[1],
        weights=[weights],
        trainable=True,
    )
    return layer
