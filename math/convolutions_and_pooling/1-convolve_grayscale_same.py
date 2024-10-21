#!/usr/bin/env python3
"""1. Same Convolution"""
import numpy as np


def convolve_grayscale_same(images, kernel):
    """flmefvlù"""
    m, h, w = images.shape
    kh, kw = kernel.shape
    # output_h = h AND output_w = w
    output = np.zeros((m, h, w))
    p_height = kh // 2
    p_weight = kw // 2
    padded_images = np.pad(images,
                           pad_width=((0,),
                                      (p_height,),
                                      (p_weight,)))

    for i in range(h):
        for j in range(w):
            output[:, i, j] = np.sum(padded_images[:, i:i+kh, j:j+kw] * kernel,
                                     axis=(1, 2))

    return output
