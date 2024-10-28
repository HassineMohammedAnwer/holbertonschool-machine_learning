#!/usr/bin/env python3
"""0. Convolutional Forward Prop"""
import numpy as np


def conv_forward(A_prev, W, b, activation,
                 padding="same", stride=(1, 1)):
    """performs forward propagation over a convolutional layer of nn:
    A_prev numpy.ndarray(m,h_prev,w_prev,c_prev): output of previous layer
    m is the number of examples
    h_prev is the height of the previous layer
    w_prev is the width of the previous layer
    c_prev is the number of channels in the previous layer
    W numpy.ndarray(kh, kw, c_prev, c_new)containing kernels for convolution
    kh is the filter height
    kw is the filter width
    c_prev is the number of channels in the previous layer
    c_new is the number of channels in the output
    bnumpy.ndarray(1, 1, 1, c_new) containing the biases applied to the convolution
    activation is an activation function applied to the convolution
    padding is a string that is either same or valid, indicating the type of padding used
    stride is a tuple of (sh, sw) containing the strides for the convolution
    sh is the stride for the height
    sw is the stride for the width
    you may import numpy as np
    Returns: the output of the convolutional layer
    """
    m, h_prev, w_prev, c_prev = A_prev.shape
    kh, kw, c_prev, c_new = W.shape
    sh, sw = stride
    if padding == 'same':
        output_h = h_prev
        output_w = w_prev
        p_top = p_bot = int(((h_prev - 1) * sh + kh - h_prev) / 2) + 1
        p_left = p_right = int(((w_prev - 1) * sw + kw - w_prev) / 2) + 1
    elif padding == 'valid':
        output_h = (h_prev - kh) // sh + 1
        output_w = (w_prev - kw) // sw + 1
        p_top, p_bot, p_left, p_right = (0, 0, 0, 0)
    padded_images = np.pad(A_prev,
                           pad_width=((0, 0),
                                      (p_top, p_bot),
                                      (p_left, p_right),
                                      (0, 0)),
                           mode='constant',
                           constant_values=0)

    A = np.zeros((m, output_h, output_w, c_new))
    for i in range(output_h):
        for j in range(output_w):
            for k in range(c_new):
                A[:, i, j, k] = np.sum(
                    padded_images[:, i * sh:i * sh + kh, j * sw:j * sw + kw, :]
                    * W[:, :, :, k], axis=(1, 2, 3))
    Z = A + b
    return Z
