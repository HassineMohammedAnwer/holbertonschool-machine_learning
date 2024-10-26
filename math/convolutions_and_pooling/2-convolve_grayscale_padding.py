#!/usr/bin/env python3
"""2. Convolution with Padding"""
import numpy as np


def convolve_grayscale_padding(images, kernel, padding):
    """performs a convolution on grayscale images with custom padding
    images is a numpy.ndarray (m, h, w) containing multiple grayscale images
    m number of images
    h height in pixels of the images
    w width in pixels of the images
    kernel numpy.ndarray (kh, kw) containing kernel for the convolution
    kh is the height of the kernel
    kw is the width of the kernel
    padding is a tuple of (ph, pw)
    ph is the padding for the height of the image
    pw is the padding for the width of the image
    the image should be padded with 0’s
    You are only allowed to use two for loops;
    Returns: a numpy.ndarray containing the convolved images"""
    m, h, w = images.shape
    kh, kw = kernel.shape
    ph, pw = padding
    output_h = h - kh + ph * 2 + 1
    output_w = w - kw + pw * 2 + 1
    output = np.zeros((m, output_h, output_w))
    padded_images = np.pad(images,
                           pad_width=((0,),
                                      (ph,),
                                      (pw,),),
                           mode='constant')
    for i in range(output_h):
        for j in range(output_w):
            output[:, i, j] = np.sum(padded_images[:, i:i+kh, j:j+kw] * kernel,
                                     axis=(1, 2))

    return output
