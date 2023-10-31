#!/usr/bin/env python3
""" function thatConvol Forward Prop """
import numpy as np


def pool_forward(A_prev, kernel_shape, stride=(1, 1), mode='max'):
    """
    forward propagation over a pooling layer
    - A_prev: shape (m, h_prev, w_prev, c_prev):utput of previous layer.
    - kernel_shape: tuple of (kh, kw): size of pooling kernel.
    - stride: a tuple of (sh, sw): strides for pooling operation.
    - mode: string: type of pooling,'max' or 'avg'.
    - The output of pooling layer.
    """
    m, h_prev, w_prev, c_prev = A_prev.shape
    kh, kw = kernel_shape
    sh, sw = stride

    # Calculate dimensions of the output feature map
    out_h = int((h_prev - kh) / sh + 1)
    out_w = int((w_prev - kw) / sw + 1)

    # Initialize the result array to store the output feature map
    result = np.zeros((m, out_h, out_w, c_prev))

    for i in range(out_h):
        for j in range(out_w):
            for k in range(c_prev):
                # Extract a slice of the input feature map
                input_slice = A_prev[:, i * sh:i * sh + kh, j * sw:j * sw + kw, k]

                # Apply pooling based on the mode
                if mode == 'max':
                    result[:, i, j, k] = np.max(input_slice, axis=(1, 2))
                elif mode == 'avg':
                    result[:, i, j, k] = np.mean(input_slice, axis=(1, 2))

    return result
