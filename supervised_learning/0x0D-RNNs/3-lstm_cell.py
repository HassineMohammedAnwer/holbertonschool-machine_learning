#!/usr/bin/env python3
"""LSTM unit"""
import numpy as np


def softmax(x):
    """The activation function"""
    tmp = np.exp(x)
    y = np.exp(x) / np.sum(tmp, axis=1, keepdims=True)
    return y


def sigmoid(x):
    """activation function"""
    return 1 / (1 + np.power(np.e, -x))


class LSTMCell:
    """dlsvcsvmsl;vs"""
    def __init__(self, i, h, o):
        """ weights and biases initializing"""
        """Wf and bf are for the forget gate"""
        self.Wf = np.random.normal(size=(i + h, h))
        self.bf = np.zeros((1, h))
        """Wu and bu are for the update gate"""
        self.Wu = np.random.normal(size=(i + h, h))
        self.bu = np.zeros((1, h))
        """Wcand bc are for the intermediate cell state"""
        self.Wc = np.random.normal(size=(i + h, h))
        self.bc = np.zeros((1, h))
        """Woand bo are for the output gate"""
        self.Wo = np.random.randn(i + h, h)
        self.bo = np.zeros((1, h))
        """Wyand by are for the output layer"""
        self.Wy = np.random.normal(size=(h, o))
        self.by = np.zeros((1, o))

    def forward(self, h_prev, c_prev, x_t):
        """forward propagation for one time step"""
        comb = np.concatenate((h_prev, x_t), axis=1)
        forget_gate = sigmoid((comb @ self.Wf) + self.bf)
        update_gate = sigmoid((comb @ self.Wu) + self.bu)
        c_candidate = np.tanh((comb @ self.Wc) + self.bc)
        c_next = forget_gate * c_prev + update_gate * c_candidate
        output_gate = sigmoid((comb @ self.Wo) + self.bo)
        h_next = output_gate * np.tanh(c_next)

        y = softmax((h_next @ self.Wy) + self.by)

        return h_next, c_next, y
