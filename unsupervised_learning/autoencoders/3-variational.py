#!/usr/bin/env python3
"""\"Variational\" Autoencoder"""

import tensorflow.keras as keras


def sampling(args):
    """Reparameterization trick: z = mu + sigma * epsilon"""
    mu, log_sigma = args
    batch = keras.backend.shape(mu)[0]
    dim = keras.backend.shape(mu)[1]
    epsilon = keras.backend.random_normal(shape=(batch, dim))
    return mu + keras.backend.exp(log_sigma / 2) * epsilon

def autoencoder(input_dims, hidden_layers, latent_dims):
    """
    creates a Variational autoencoder, model should be compiled using adam
    optimization and binary cross-entropy loss.All layers should use a relu activation
    except for the mean and log variance layers in the encoder, which should use None,
    and the last layer in the decoder, which should use sigmoid
    Args:
        input_dims: integer containing the dimensions of the model input
        hidden_layers: list containing the number of nodes for each hidden layer in
        the encoder, respectively the hidden layers should be reversed for the decoder
        latent_dims: integer containing the dimensions of the latent space
                      representation
    Returns: encoder, decoder, auto
            encoder: encoder model
            decoder: decoder model
            auto: the full autoencoder model
    """
    x = keras.Input(shape=(input_dims,))
    h_l = keras.layers.Dense(units=hidden_layers[0], activation='relu')
    Y_prev = h_l(x)
    for i in range(1, len(hidden_layers)):
        h_l = keras.layers.Dense(units=hidden_layers[i],
                                       activation='relu')
        Y_prev = h_l(Y_prev)
    latent = keras.layers.Dense(units=latent_dims, activation=None)
    z_mean = latent(Y_prev)
    z_log_sigma = latent(Y_prev)

    z = keras.layers.Lambda(sampling,
                            output_shape=(latent_dims,))([z_mean, z_log_sigma])
    encoder = keras.Model(x, [z, z_mean, z_log_sigma])
    x_decode = keras.Input(shape=(latent_dims,))
    h_l_deco = keras.layers.Dense(units=hidden_layers[-1],
                                        activation='relu')
    Y_prev = h_l_deco(x_decode)
    for j in range(len(hidden_layers) - 2, -1, -1):
        h_l_deco = keras.layers.Dense(units=hidden_layers[j],
                                            activation='relu')
        Y_prev = h_l_deco(Y_prev)
    last_l = keras.layers.Dense(units=input_dims, activation='sigmoid')
    output = last_l(Y_prev)
    decoder = keras.Model(x_decode, output)
    e_output = encoder(x)[-1]
    d_output = decoder(e_output)
    auto = keras.Model(x, d_output)

    auto.compile(loss='binary_crossentropy', optimizer='adam')
    return encoder, decoder, auto
