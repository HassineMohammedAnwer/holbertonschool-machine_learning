#!/usr/bin/env python3
"""6. Pooling"""
import numpy as np


def pool(images, kernel_shape, stride, mode='max'):
    """ performs pooling on images:
    images numpy.ndarray (m, h, w, c) containing multiple images
    m is the number of images
    h is the height in pixels of the images
    w is the width in pixels of the images
    c is the number of channels in the image
    kernel_shape tuple of (kh, kw) containing kernel shape for pooling
    kh is the height of the kernel
    kw is the width of the kernel
    stride is a tuple of (sh, sw)
    sh is the stride for the height of the image
    sw is the stride for the width of the image
    mode indicates the type of pooling
    max indicates max pooling
    avg indicates average pooling
    You are only allowed to use two for loops
    Returns: a numpy.ndarray containing the pooled images"""
    m, h, w, c = images.shape
    kh, kw = kernel_shape
    sh, sw = stride
    output_h = (h - kh) // sh + 1
    output_w = (w - kw) // sw + 1
    output = np.zeros((m, output_h, output_w, c))
    for i in range(output_h):
        for j in range(output_w):
            if mode == "max":
                output[:, i, j, :] = np.max(
                    images[:, i * sh:i * sh + kh, j * sw:j * sw + kw],
                    axis=(1, 2)
                )
            elif mode == 'avg':
                output[:, i, j, :] = np.average(
                    images[:, i * sh:i * sh + kh, j * sw:j * sw + kw],
                    axis=(1, 2)
                )
    return output
