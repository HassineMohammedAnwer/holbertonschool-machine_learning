#!/usr/bin/env python3
"""0. Initialize"""
import numpy as np
import tensorflow as tf


class NST:
    """ performs tasks for neural style transfer"""
    style_layers = ['block1_conv1', 'block2_conv1', 'block3_conv1', 'block4_conv1', 'block5_conv1']
    content_layer = 'block5_conv2'

    def __init__(self, style_image, content_image, alpha=1e4, beta=1):
        """Class constructo"""
        if not isinstance(style_image, np.ndarray) or style_image.shape[2] != 3:
            raise TypeError("style_image must be a numpy.ndarray with shape (h, w, 3)")
        if not isinstance(content_image, np.ndarray) or content_image.shape[2] != 3:
            raise TypeError("content_image must be a numpy.ndarray with shape (h, w, 3)")
        if not isinstance(alpha, (int, float)) or alpha < 0:
            raise TypeError("alpha must be a non-negative number")
        if not isinstance(beta, (int, float)) or beta < 0:
            raise TypeError("beta must be a non-negative number")
        self.style_image = self.scale_image(style_image)
        self.content_image = self.scale_image(content_image)
        self.alpha = alpha
        self.beta = beta

    @staticmethod
    def scale_image(image):
        """ rescales an image such that its pixels values are between 0
        and 1 and its largest side is 512 pixels"""
        if not isinstance(image, np.ndarray) or image.shape[2] != 3:
            raise TypeError("image must be a numpy.ndarray with shape (h, w, 3)")
        h, w, _ = image.shape
        max_dim = max(h, w)
        scale_factor = 512 / max_dim
        new_h, new_w = int(h * scale_factor), int(w * scale_factor)
        resized_img = tf.image.resize(image, [new_h, new_w], method='bicubic')
        scaled_img = resized_img / 255.0
        scaled_img = tf.expand_dims(scaled_img, axis=0)

        return scaled_img