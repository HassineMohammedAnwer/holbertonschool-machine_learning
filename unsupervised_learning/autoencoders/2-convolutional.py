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
    x = keras.Input(shape=input_dims)

    encoder_conv = keras.layers.Conv2D(filters=filters[0], kernel_size=(3, 3),
                                       padding='same',
                                       activation='relu')(x)

    encoder_pool = keras.layers.MaxPooling2D(pool_size=(2, 2),
                                             padding="same")(encoder_conv)

    for i in range(1, len(filters)):
        encoder_conv = keras.layers.Conv2D(filters=filters[i],
                                           kernel_size=(3, 3), padding='same',
                                           activation='relu')(encoder_pool)
        encoder_pool = keras.layers.MaxPooling2D(pool_size=(2, 2),
                                                 padding="same")(encoder_conv)

    latent_ly = encoder_pool
    encoder = keras.Model(x, latent_ly)

    X_decoder = keras.Input(shape=latent_dims)
    decoder_conv = keras.layers.Conv2D(filters=filters[-1],
                                       kernel_size=(3, 3),
                                       padding='same',
                                       activation='relu')(X_decoder)

    decoder_pool = keras.layers.UpSampling2D((2, 2))(decoder_conv)

    for j in range(len(filters) - 2, 0, -1):
        decoder_conv = keras.layers.Conv2D(filters=filters[j],
                                           kernel_size=(3, 3),
                                           padding='same',
                                           activation='relu')(decoder_pool)
        decoder_pool = keras.layers.UpSampling2D((2, 2))(decoder_conv)

    decoder_conv = keras.layers.Conv2D(filters=filters[0], kernel_size=(3, 3),
                                       padding='valid',
                                       activation='relu')(decoder_pool)

    decoder_pool = keras.layers.UpSampling2D((2, 2))(decoder_conv)

    output = keras.layers.Conv2D(filters=input_dims[-1], kernel_size=(3, 3),
                                 padding='same',
                                 activation='sigmoid')(decoder_pool)

    decoder = keras.Model(X_decoder, output)

    X_input = keras.Input(shape=input_dims)
    e_output = encoder(X_input)
    d_output = decoder(e_output)
    auto = keras.Model(inputs=X_input, outputs=d_output)
    auto.compile(loss='binary_crossentropy', optimizer='adam')
    return encoder, decoder, auto
