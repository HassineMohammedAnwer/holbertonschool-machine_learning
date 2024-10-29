#!/usr/bin/env python3
"""5. LeNet-5 (Keras)"""
from tensorflow import keras as K


def lenet5(x):
    """ builds a modified version of the LeNet-5 architecture using keras:
    X is a K.Input(m, 28, 28, 1): the input images for the network
    m is the number of images
    The model should consist of the following layers in order:
    Convolutional layer with 6 kernels of shape 5x5 with same padding
    Max pooling layer with kernels of shape 2x2 with 2x2 strides
    Convolutional layer with 16 kernels of shape 5x5 with valid padding
    Max pooling layer with kernels of shape 2x2 with 2x2 strides
    Fully connected layer with 120 nodes
    Fully connected layer with 84 nodes
    Fully connected softmax output layer with 10 nodes
    All layers requiring initialization should initialize their kernels
    __with the he_normal initialization method
    The seed for the he_normal initializer should be set to zero for each layer
    __to ensure reproducibility.
    All hidden layers requiring activation should use relu activation function
    you may from tensorflow import keras as K
    Returns: a K.Model compiled to use Adam optimization
    __(with default hyperparameters) and accuracy metrics
    """
    initializer = K.initializers.HeNormal(seed=0)
    model = K.Sequential([
        K.layers.Conv2D(filters=6,
                        kernel_size=5,
                        padding='same',
                        kernel_initializer=initializer,
                        activation='relu'),
        K.layers.MaxPooling2D(pool_size=2,
                              strides=2),
        K.layers.Conv2D(filters=16,
                        kernel_size=5,
                        padding='valid',
                        kernel_initializer=initializer,
                        activation='relu'),
        K.layers.MaxPooling2D(pool_size=2,
                              strides=2),
        K.layers.Flatten(),
        K.layers.Dense(120, activation='relu', kernel_initializer=initializer),
        K.layers.Dense(84, activation='relu', kernel_initializer=initializer),
        K.layers.Dense(10, activation='softmax',
                       kernel_initializer=initializer)
    ])
    model.compile(loss='categorical_crossentropy',
                  optimizer=K.optimizers.Adam(),
                  metrics=['accuracy'])
    return model
