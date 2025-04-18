#!/usr/bin/env python3
"""8. Save Only the Best"""
import tensorflow.keras as K


def train_model(network, data, labels, batch_size, epochs,
                validation_data=None, early_stopping=False, patience=0,
                learning_rate_decay=False, alpha=0.1, decay_rate=1,
                save_best=False, filepath=None, verbose=True, shuffle=False):
    """
    trains model using mini-bch gradient descent and analyzes validation data
      also trains the model using early stopping also train the model with
      learning rate decay-_-also save the best iteration of the model
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
    learning_rate_decay boolean indicates whether to use learning rate decay
    alpha is the initial learning rate
    decay_rate is the decay rate
    -_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-__-_-_-_-_-_
    save_best boolean indicating whether to save model after each epoch if it's
    the best
       a model is considered the best if its validation loss is the lowest
       that the model has obtained
    filepath is the file path where the model should be saved
    Returns: the History object generated after training the model"""
    callbacks = []
    if early_stopping is True and validation_data is not None:
        callback_early_stop = K.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=patience,
            verbose=verbose,
        )
        callbacks.append(callback_early_stop)
    if learning_rate_decay and validation_data:
        def lr_decay(epoch, lr):
            return alpha / (1 + decay_rate * epoch)
        callback_l_r_d = K.callbacks.LearningRateScheduler(
            schedule=lr_decay, verbose=1)
        callbacks.append(callback_l_r_d)
        # save best model
    if save_best:
        callback_save_best = K.callbacks.ModelCheckpoint(
            filepath=filepath,
            monitor='val_loss',
            save_best_only=True
        )

        callbacks.append(callback_save_best)

    History = network.fit(
        data,
        labels,
        batch_size=batch_size,
        verbose=verbose,
        callbacks=callbacks,
        validation_data=validation_data,
        epochs=epochs,
        shuffle=shuffle)
    return History
