#!/usr/bin/env python3
"""12. Test"""
import tensorflow.keras as K


def test_model(network, data, labels, verbose=True):
    """ that tests a neural network:
    network is the network model to test
    data is the input data to test the model with
    labels are the correct one-hot labels of data
    verbose boolean determines if output should be printed
      during testing process
    Returns: loss and accuracy of model with testing data,respectively"""
    return network.evaluate(x=data,
                            y=labels,
                            verbose=verbose)
