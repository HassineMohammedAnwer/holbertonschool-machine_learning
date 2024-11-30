#!/usr/bin/env python3
"""1. Crop"""

import tensorflow as tf


def crop_image(image, size):
    """ performs a random crop of an image:
    image is a 3D tf.Tensor containing the image to crop
    size is a tuple containing the size of the crop
    Returns the cropped image"""
    return tf.image.random_crop(
        value=image, size=size, seed=None, name=None
    )
