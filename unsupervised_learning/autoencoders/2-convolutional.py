#!/usr/bin/env python3
"""\"convolutional\" Autoencoder"""

import tensorflow.keras as keras


def autoencoder(input_dims, filters, latent_dims):
    """
    creates a convolutional autoencoder, model should be compiled using adam
    optimization and binary cross-entropy loss.
    Args:
        input_dims: integer containing the dimensions of the model input
        filters: list, number of filters for each convolutional layer in the encoder
        latent_dims: integer containing the dimensions of the latent space
                      representation
    Returns: encoder, decoder, auto
            encoder: encoder model
            decoder: decoder model
            auto: the full autoencoder model
    """

