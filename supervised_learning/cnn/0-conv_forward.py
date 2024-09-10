#!/usr/bin/env python3
""" function thatConvol Forward Prop """
import numpy as np


def conv_forward(A_prev, W, b, activation,
                 padding="same", stride=(1, 1)):
    """ A_prev: shape (m, h_prev, w_prev, c_prev): output of the previous layer.
    W: shape (kh, kw, c_prev, c_new)  kernels for the convolution
    b: shape (1, 1, 1, c_new) biases
    activation: Activation function to be applied to the convolutional result
    padding: String that can be either "same" or "valid," indicating the type of padding used.
    """
    m, h_prev, w_prev, c_prev = A_prev.shape
    kh, kw, c_prev, c_new = W.shape
    sh, sw = stride
    # Padding the input according to the 'same' padding option
    if padding == "same":
        ph = int(((h_prev - 1) * sh + kh - h_prev) / 2)
        pw = int(((w_prev - 1) * sw + kw - w_prev) / 2)
        A_prev = np.pad(A_prev, ((0, 0), (ph, ph), (pw, pw), (0, 0)), mode='constant')
    # Calculate the dimensions of the output feature map
    h_output = int((h_prev - kh) / sh + 1)
    w_output = int((w_prev - kw) / sw + 1)
    A = np.zeros((m, h_output, w_output, c_new))
    # Perform the convolution operation
    for i in range(h_output):
        for j in range(w_output):
            for k in range(c_new):
                # Convolve each filter (k-th) at position (i, j)
                A[:, i, j, k] = np.sum(A_prev[:, i * sh:i * sh + kh, j * sw:j * sw + kw, :] * W[:, :, :, k], axis=(1, 2, 3))

    Z = A + b
    return activation(Z)
