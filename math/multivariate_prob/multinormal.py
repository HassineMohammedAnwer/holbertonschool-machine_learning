#!/usr/bin/env python3
"""3. PDF"""
import numpy as np


class MultiNormal:
    """Multivariate Normal distribution"""
    def __init__(self, data):
        """Initialize a Multivariate Normal distribution."""
        if not isinstance(data, np.ndarray) or len(data.shape) != 2:
            raise TypeError("data must be a 2D numpy.ndarray")
        d, n = data.shape
        if n < 2:
            raise ValueError("data must contain multiple data points")
        self.mean = np.mean(data, axis=1, keepdims=True)
        data_centered = data - self.mean
        self.cov = np.dot(data_centered, data_centered.T) / (n - 1)

    def pdf(self, x):
        """Probability Density Function
        f(x) = (1 / (sqrt((2 * π)^d * det(Σ)))) *
              exp(-0.5 * (x - μ)^T * Σ^(-1) * (x - μ))
        d is the number of dimensions (i.e., the size of the mean vector μ)
        Σ is the covariance matrix of the distribution
        det(Σ) determinant of the covariance matrix
        Σ^(-1) is the inverse of the covariance matrix

        """
        if not isinstance(x, np.ndarray):
            raise TypeError("x must be a numpy.ndarray")
        d = self.mean.shape[0]
        if x.shape != (d, 1):
            raise ValueError("x must have the shape ({}, 1)".
                             format(d))
        tmp = x - self.mean
        cov_det = np.linalg.det(self.cov)
        cov_inv = np.linalg.inv(self.cov)
        tmp2 = np.dot(tmp.T, cov_inv)
        exponent = -0.5 * np.dot(tmp2, tmp)
        coefficient = (1 / np.sqrt((2 * np.pi) ** d * cov_det))
        pdf_value = coefficient * np.exp(exponent)
        return pdf_value[0][0]
