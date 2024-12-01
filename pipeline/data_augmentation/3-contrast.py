#!/usr/bin/env python3
"""3. Contrast"""

import tensorflow as tf


def change_contrast(image, lower, upper):
    """randomly adjusts the contrast of an image.
    image: A 3D tf.Tensor representing the input image to adjust the contrast.
    lower: A float representing the lower bound of the random contrast factor range.
    upper: A float representing the upper bound of the random contrast factor range.
    Returns the contrast-adjusted image."""
    image = tf.image.convert_image_dtype(image, tf.float32)
    seed = (0, 0)
    contrast_image =  tf.image.stateless_random_contrast(
        image=image, lower=lower, upper=upper, seed=seed
    )
    return contrast_image
