#!/usr/bin/env python3
"""6. Early Stopping"""
import tensorflow.keras as K


def train_model(network, data, labels, batch_size, epochs,
                validation_data=None, early_stopping=False, patience=0,
                learning_rate_decay=False, alpha=0.1, decay_rate=1,
                verbose=True, shuffle=False):
    """
    trains model using mini-bch gradient descent and analyzes validation data
      also trains the model using early stopping
    network is the model to train
    data is a numpy.ndarray of shape (m, nx) containing the input data
    labels one-hot numpy.ndarray of shape(m, classes)containing labels of data
    batch_size is the size of the batch used for mini-batch gradient descent
    epochs is the number of passes through data for mini-batch gradient descent
    verbose boolean determines if output should be printed during training
    shuffle boolean determines whether to shuffle batches every epoch.Normally,
      it's a good idea to shuffle,but for reproducibility,we set default=False.
    validation_data is the data to validate the model with, if not None
    early_stopping boolean that indicates whether early stopping should be used
      early stopping should only be performed if validation_data exists
      early stopping should be based on validation loss
    patience is the patience used for early stopping
    Returns: the History object generated after training the model"""
    callback = None
    if validation_data:
        callbacks = []
        if early_stopping == True:
            callback_early_stop = K.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=patience,
                verbose=verbose,
            )
        callbacks.append(callback_early_stop)
        if learning_rate_decay == True:
            l_r_d = alpha / (1 + decay_rate * epochs)
            callback_l_r_d = K.callbacks.LearningRateScheduler(
                schedule = l_r_d, verbose=1)
            callbacks.append(callback_l_r_d)

    History = network.fit(
        data,
        labels,
        batch_size=batch_size,
        verbose=verbose,
        callbacks=callback,
        validation_data=validation_data,
        epochs=epochs,
        shuffle=shuffle)
    return History
