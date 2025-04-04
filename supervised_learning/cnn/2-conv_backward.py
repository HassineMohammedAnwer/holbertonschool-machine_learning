#!/usr/bin/env python3
"""2. Convolutional Back Prop"""
import numpy as np


def conv_backward(dZ, A_prev, W, b, padding="same", stride=(1, 1)):
    """performs back propagation over a convolutional layer of NN:
    dZ is a numpy.ndarray (m, h_new, w_new, c_new): partial derivatives
    __with respect to the unactivated output of the convolutional layer
    m is the number of examples
    h_new is the height of the output
    w_new is the width of the output
    c_new is the number of channels in the output
    A_prev numpy.ndarray (m, h_prev, w_prev, c_prev): output of previous layer
    h_prev is the height of the previous layer
    w_prev is the width of the previous layer
    c_prev is the number of channels in the previous layer
    W numpy.ndarray (kh, kw, c_prev, c_new) : kernels for convolution
    kh is the filter height
    kw is the filter width
    b is a numpy.ndarray (1, 1, 1, c_new): biases applied to the convolution
    padding string that's either same or valid, indicating type of padding used
    stride is a tuple of (sh, sw) containing the strides for the convolution
    sh is the stride for the height
    sw is the stride for the width
    you may import numpy as np
    Returns: partial derivatives with respect to previous layer (dA_prev),
    __the kernels (dW), and the biases (db), respectively
    """
    m, h_new, w_new, c_new = dZ.shape
    _, h_prev, w_prev, c_prev = A_prev.shape
    kh, kw, _, _ = W.shape
    sh, sw = stride
    if padding == 'same':
        ph = int(((h_prev - 1) * sh + kh - h_prev) / 2 + 0.5)
        pw = int(((w_prev - 1) * sw + kw - w_prev) / 2 + 0.5)
        A_prev = np.pad(A_prev,
                        pad_width=((0, 0),
                                   (ph, ph),
                                   (pw, pw),
                                   (0, 0)),
                        mode='constant',
                        constant_values=0)
    else:
        ph, pw = 0, 0
    dA_prev = np.zeros((A_prev.shape))
    dW = np.zeros((kh, kw, c_prev, c_new))
    db = np.sum(dZ, axis=(0, 1, 2), keepdims=True)
    for e in range(m):
        for i in range(h_new):
            for j in range(w_new):
                for k in range(c_new):
                    dA_prev[e, i * sh:i * sh + kh,
                            j * sw:j * sw + kw, :] += (W[:, :, :, k] *
                                                       dZ[e, i, j, k])
                    dW[:, :, :, k] += (A_prev[e, i * sh:i * sh + kh,
                                              j * sw:j * sw + kw, :] *
                                       dZ[e, i, j, k])
    if padding == "same":
        dA = dA_prev[:, ph:-ph, pw:-pw, :]
    else:
        dA = dA_prev
    return dA, dW, db
