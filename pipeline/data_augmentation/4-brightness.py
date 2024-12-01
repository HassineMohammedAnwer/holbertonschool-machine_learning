#!/usr/bin/env python3
"""4. Brightness"""

import tensorflow as tf


def change_brightness(image, max_delta):
    """ randomly changes the brightness of an image:
    image is a 3D tf.Tensor containing the image to change
    max_delta the maximum amount the image should be brightened(or darkened)
    Returns the altered image"""
    return tf.image.random_brightness(
        image, max_delta)
