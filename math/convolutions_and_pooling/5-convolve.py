#!/usr/bin/env python3
"""5. Multiple Kernels"""
import numpy as np


def convolve(images, kernels, padding='same', stride=(1, 1)):
    """performs a convolution on images using multiple kernels:
    images numpy.ndarray (m, h, w, c) containing multiple images
    m is the number of images
    h is the height in pixels of the images
    w is the width in pixels of the images
    c is the number of channels in the image
    kernels numpy.ndarray (kh, kw, c, nc) containing kernels for convolution
    kh is the height of a kernel
    kw is the width of a kernel
    nc is the number of kernels
    padding is either a tuple of (ph, pw), ‘same’, or ‘valid’
    if ‘same’, performs a same convolution
    if ‘valid’, performs a valid convolution
    if a tuple:
    ph is the padding for the height of the image
    pw is the padding for the width of the image
    the image should be padded with 0’s
    stride is a tuple of (sh, sw)
    sh is the stride for the height of the image
    sw is the stride for the width of the image
    Only allowed to use three for loops;
    Returns: a numpy.ndarray containing the convolved images"""
    m, h, w, c = images.shape
    kh, kw, c, nc = kernels.shape
    sh, sw = stride
    if padding == 'same':
        output_h = h
        output_w = w
        p_top = p_bot = int(((h - 1) * sh + kh - h) / 2) + 1
        p_left = p_right = int(((w - 1) * sw + kw - w) / 2) + 1
    elif padding == 'valid':
        output_h = (h - kh) // sh + 1
        output_w = (w - kw) // sw + 1
        p_top, p_bot, p_left, p_right = (0, 0, 0, 0)
    elif isinstance(padding, tuple):
        p_h, p_w = padding
        p_top = p_h
        p_bot = p_h
        p_right = p_w
        p_left = p_w
        output_h = (h - kh + 2 * p_h) // sh + 1
        output_w = (w - kw + 2 * p_w) // sw + 1
    padded_images = np.pad(images,
                           pad_width=((0, 0),
                                      (p_top, p_bot),
                                      (p_left, p_right),
                                      (0, 0)),
                           mode='constant',
                           constant_values=0)

    output = np.zeros((m, output_h, output_w, c))

    for i in range(output_h):
        for j in range(output_w):
            for k in range(nc):
                output[:, i, j, k] = np.sum(
                    padded_images[:, i * sh:i * sh + kh, j * sw:j * sw + kw]
                    * kernels[:, :, :, k], axis=(1, 2, 3))

    return output
