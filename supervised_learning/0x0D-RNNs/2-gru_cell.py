#!/usr/bin/env python3
"""a gated recurrent unit"""
import numpy as np


def softmax(x):
    """The activation function"""
    tmp = np.exp(x)
    y = np.exp(x) / np.sum(tmp, axis=1, keepdims=True)
    return y


def sigmoid(x):
    """activation function"""
    return 1 / (1 + np.power(np.e, -x))


class GRUCell:
    """only two gates:update gate (z_t) and reset gate (r_t)"""
    def __init__(self, i, h, o):
        """ weights and biases initializing"""
        """Wz and bz are for the update gate"""
        self.Wz = np.random.normal(size=(i + h, h))
        self.bz = np.zeros((1, h))
        """Wr and br are for the reset gate"""
        self.Wr = np.random.normal(size=(i + h, h))
        self.br = np.zeros((1, h))
        """Whand bh are for the intermediate hidden state"""
        self.Wh = np.random.normal(size=(i + h, h))
        self.bh = np.zeros((1, h))
        """Wyand by are for the output"""
        self.Wy = np.random.normal(size=(h, o))
        self.by = np.zeros((1, o))

    def forward(self, h_prev, x_t):
        """forward propagation for one time step"""
        comb1= np.concatenate((h_prev, x_t), axis=1)
        """update gate"""
        z_t = sigmoid((comb1 @ self.Wz) + self.bz)
        """reset gate"""
        r_t = sigmoid((comb1 @ self.Wr) + self.br)
        comb2 = np.concatenate(((r_t * h_prev), x_t), axis=1)
        condidate_h = np.tanh(comb2 @ self.Wh + self.bh)

        s_t = (1 - z_t) * h_prev + z_t * condidate_h

        y = softmax((s_t @ self.Wy) + self.by)
        return s_t, y

