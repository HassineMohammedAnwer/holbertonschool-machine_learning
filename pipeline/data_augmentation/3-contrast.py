#!/usr/bin/env python3
"""3. Contrast"""

import tensorflow as tf


def change_contrast(image, lower, upper):
    """ rotates an image by 90 degrees counter-clockwise:
    image is a 3D tf.Tensor containing the image to rotate
    Returns the rotated image"""
    return tf.image.random_contrast(
        image, lower, upper)
