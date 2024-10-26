#!/usr/bin/env python3
"""3. Strided Convolution"""
import numpy as np


def convolve_grayscale(images, kernel, padding='same', stride=(1, 1)):
    """performs a convolution on grayscale images:
    images numpy.ndarray (m, h, w) containing multiple grayscale images
    m is the number of images
    h is the height in pixels of the images
    w is the width in pixels of the images
    kernel is a numpy.ndarray (kh, kw) containing kernel for convolution
    kh is the height of the kernel
    kw is the width of the kernel
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
    You are only allowed to use two for loops;  Hint: loop over i and j
    Returns: a numpy.ndarray containing the convolved images"""
    m, h, w = images.shape
    kh, kw = kernel.shape
    sh, sw = stride
    if padding == 'same':
        output_h = h
        output_w = w
        p_top = p_bot = ((h - 1) * sh + kh - h) // 2
        p_left = p_right = ((w - 1) * sw + kw - w) // 2
    elif padding == 'valid':
        output_h = (h - kh) // sh + 1
        output_w = (w - kw) // sw + 1
        p_top, p_bot, p_left, p_right = (0, 0, 0, 0)
    elif isinstance(padding, tuple):
        p_top = p_bot , p_left = p_right = padding
        output_h = (h - kh + 2 * p_top) // sh + 1
        output_w = (w - kw + 2 * p_left) // sw + 1
    padded_images = np.pad(images,
                           pad_width=((0, 0),
                                       (p_top, p_bot),
                                       (p_left, p_right)),
                           mode='constant',
                           constant_values=0)

    output = np.zeros((m, output_h, output_w))

    for i in range(output_h):
        for j in range(output_w):
            output[:, i, j] = np.sum(padded_images[:, i * sh:i * sh + kh, j * sw:j * sw + kw]
                                     * kernel, axis=(1, 2))

    return output
