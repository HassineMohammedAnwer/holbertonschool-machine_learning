#!/usr/bin/env python3
"""4. Train"""
import tensorflow.keras as K


def train_model(network, data, labels, batch_size,
                epochs, verbose=True, shuffle=False):
    """trains a model using mini-batch gradient descent:
    network is the model to train
    data is a numpy.ndarray of shape (m, nx) containing the input data
    labels one-hot numpy.ndarray of shape(m, classes)containing labels of data
    batch_size is the size of the batch used for mini-batch gradient descent
    epochs is the number of passes through data for mini-batch gradient descent
    verbose boolean determines if output should be printed during training
    shuffle boolean determines whether to shuffle batches every epoch.Normally,
      it's a good idea to shuffle,but for reproducibility,we set default=False.
    Returns: the History object generated after training the model"""
    History = network.fit(
        data,
        labels,
        batch_size=batch_size,
        verbose=verbose,
        nb_epoch=epochs,
        shuffle=shuffle)
    return History
