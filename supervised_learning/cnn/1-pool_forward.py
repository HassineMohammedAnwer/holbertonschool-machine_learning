#!/usr/bin/env python3
"""0. Convolutional Forward Prop"""
import numpy as np


def pool_forward(A_prev, kernel_shape, stride=(1, 1), mode='max'):
    """performs forward propagation over a pooling layer of a nn:
    A_prev numpy.ndarray(m, h_prev, w_prev, c_prev): output previous layer
    m is the number of examples
    h_prev is the height of the previous layer
    w_prev is the width of the previous layer
    c_prev is the number of channels in the previous layer
    kernel_shape tuple(kh, kw): the size of the kernel for the pooling
    kh is the kernel height
    kw is the kernel width
    stride is a tuple of (sh, sw) containing the strides for the pooling
    sh is the stride for the height
    sw is the stride for the width
    mode is a string containing either max or avg,
    you may import numpy as np
    Returns: the output of the pooling layer
    """
    m, h_prev, w_prev, c_prev = A_prev.shape
    kh, kw = kernel_shape
    sh, sw = stride
    output_h = (h_prev - kh) // sh + 1
    output_w = (w_prev - kw) // sw + 1
    output = np.zeros((m, output_h, output_w, c_prev))
    for i in range(output_h):
        for j in range(output_w):
            if mode == "max":
                output[:, i, j, :] = np.max(
                    A_prev[:, i * sh:i * sh + kh, j * sw:j * sw + kw],
                    axis=(1, 2)
                )
            elif mode == 'avg':
                output[:, i, j, :] = np.average(
                    A_prev[:, i * sh:i * sh + kh, j * sw:j * sw + kw],
                    axis=(1, 2)
                )
    return output
