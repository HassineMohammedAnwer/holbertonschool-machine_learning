#!/usr/bin/env python3
"""11. Save and Load Configuration"""
import tensorflow.keras as K


def save_config(network, filename):
    """ saves a model’s configuration in JSON format:
     is the model whose configuration should be saved
    filename path of the file that the configuration should be saved to
    Returns: None"""
    network_json = network.to_json()
    with open(filename, "w") as f:
        f.write(network_json)


def load_config(filename):
    """ loads a model with a specific configuration:
    filename path of file containing model’s configuration in JSON format
    Returns: the loaded model"""
    with open(filename, "r") as f:
        network_json = f.read()
    return K.models.model_from_json(network_json)
