#!/usr/bin/env python3
"""\"Sparse\" Autoencoder"""

import tensorflow.keras as keras


def autoencoder(input_dims, hidden_layers, latent_dims, lambtha):
    """
    creates a sparse  autoencoder, model should be compiled using adam optimization
    and binary cross-entropy loss. All layers should use a relu activation
    except for the last layer in the decoder, which should use sigmoid
    Args:
        input_dims: integer containing the dimensions of the model input
        hidden_layers: list containing the number of nodes for each hidden
                       layer in the encoder, respectively hidden layers should
                       be reversed for the decoder
        latent_dims: integer containing the dimensions of the latent space
                      representation
        lambtha: regularization parameter used for L1 regularization
                      on the encoded output
    Returns: encoder, decoder, auto
            encoder: encoder model
            decoder: decoder model
            auto: the full autoencoder model
    """
    inputs = keras.Input(shape=(input_dims,))
    x = inputs
    for units in hidden_layers:
        x = keras.layers.Dense(units, activation='relu')(x)
    latent = keras.layers.Dense(latent_dims, activation='relu',
                                activity_regularizer = keras.regularizers.l1(lambtha))(x)

    encoder = keras.Model(inputs, latent, name='encoder')

    latent_inputs = keras.Input(shape=(latent_dims,))
    x = latent_inputs
    for units in reversed(hidden_layers):
        x = keras.layers.Dense(units, activation='relu')(x)
    outputs = keras.layers.Dense(input_dims, activation='sigmoid')(x)

    decoder = keras.Model(latent_inputs, outputs, name='decoder')

    autoencoder_input = keras.Input(shape=(input_dims,))
    encoded = encoder(autoencoder_input)
    decoded = decoder(encoded)
    auto = keras.Model(autoencoder_input, decoded, name='sparse_autoencoder')

    auto.compile(optimizer='adam', loss='binary_crossentropy')
    return encoder, decoder, auto
