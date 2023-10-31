#!/usr/bin/env python3
""" function thatConvol Forward Prop """
import numpy as np


def conv_backward(dZ, A_prev, W, b, padding="same", stride=(1, 1)):
    """
    backpropagation
    - dZ:(m, h_new, w_new, c_new): partial derivatives respect to the unactivated output
    - A_prev (m, h_prev, w_prev, c_prev): output of previous layer
    - W:(kh, kw, c_prev, c_new): kernels
    - b:(1, 1, 1, c_new): biases
    - padding: string "same" or "valid": the type of padding
    - stride: a tuple of (sh, sw): strides
    - The partial derivatives respect to previous layer (dA_prev),
    the kernels (dW), and the biases (db).
    """

    m, h_new, w_new, c_new = dZ.shape
    kh, kw, c_prev, _ = W.shape
    sh, sw = stride

    if padding == "same":
        pad_h = int(((h_new - 1) * sh + kh - h_new) / 2)
        pad_w = int(((w_new - 1) * sw + kw - w_new) / 2)
        A_prev = np.pad(A_prev, ((0, 0), (pad_h, pad_h), (pad_w, pad_w), (0, 0)), mode='constant')

    h_prev, w_prev, c_prev = A_prev.shape[1], A_prev.shape[2], A_prev.shape[3]
    dA_prev = np.zeros_like(A_prev)
    dW = np.zeros_like(W)
    db = np.sum(dZ, axis=(0, 1, 2), keepdims=True)

    for i in range(m):
        for h in range(h_new):
            for w in range(w_new):
                for c in range(c_new):
                    dA_prev[i, h * sh: h * sh + kh, w * sw: w * sw + kw, :] += W[:, :, :, c] * dZ[i, h, w, c]
                    dW[:, :, :, c] += A_prev[i, h * sh: h * sh + kh, w * sw: w * sw + kw, :] * dZ[i, h, w, c]

    return dA_prev, dW, db
