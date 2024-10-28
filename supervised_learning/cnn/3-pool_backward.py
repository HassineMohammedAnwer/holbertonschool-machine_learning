#!/usr/bin/env python3
"""3. Pooling Back Prop"""
import numpy as np


def pool_backward(dA, A_prev, kernel_shape, stride=(1, 1), mode='max'):
    """performs back propagation over a pooling layer of a neural network:
    dA numpy.ndarray(m, h_new, w_new, c_new):partial derivatives with respect
    __to the output of the pooling layer
    m is the number of examples
    h_new is the height of the output
    w_new is the width of the output
    c is the number of channels
    A_prev numpy.ndarray(m, h_prev, w_prev, c): output of the previous layer
    h_prev is the height of the previous layer
    w_prev is the width of the previous layer
    kernel_shape is a tuple of (kh, kw): the size of the kernel for the pooling
    kh is the kernel height
    kw is the kernel width
    stride is a tuple of (sh, sw) containing the strides for the pooling
    sh is the stride for the height
    sw is the stride for the width
    mode is a string containing either max or avg
    you may import numpy as np
    Returns: partial derivatives with respect to the previous layer (dA_prev)
    """
    m, h_new, w_new, c_new = dA.shape
    _, h_prev, w_prev, c = A_prev.shape
    kh, kw, = kernel_shape
    sh, sw = stride
    dA_prev = np.zeros((A_prev.shape))
    if mode == 'avg':
        for e in range(m):
            for i in range(h_new):
                for j in range(w_new):
                    for k in range(c_new):
                        average_dA = dA[e, i, j, k] / kh / kw
                        dA_prev[e, i * sh:(i * sh + kh), j * sw:(j * sw + kw),
                                k] += np.ones((kh, kw))*average_dA
    elif mode == 'max':
        for e in range(m):
            for i in range(h_new):
                for j in range(w_new):
                    for k in range(c_new):
                        a_prev_slice = A_prev[e, i * sh:i * sh + kh,
                                              j * sw:j * sw + kw, k]
                        mask = (a_prev_slice == np.max(a_prev_slice))
                        dA_prev[e, i * sh:(i * sh + kh), j * sw:(j * sw + kw),
                                k] += mask * dA[e, i, j, k]

    return dA_prev
