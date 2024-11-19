#!/usr/bin/env python3
"""5. PDF"""

import numpy as np


def pdf(X, m, S):
    """Probability Density Function (PDF) of a Gaussian Distribution
    P(x)= 1 / ( V-(2π)**d det(cov) ) * exp( -(1/2) * (x−μ)**T cov**−1 (x−μ))
    cov = S
    """
    if not isinstance(X, np.ndarray) or X.ndim != 2:
        return None
    if not isinstance(m, np.ndarray) or m.ndim != 1:
        return None
    if not isinstance(S, np.ndarray) or S.ndim != 2:
        return None
    if (X.shape[1] != m.shape[0] or S.shape[0] != S.shape[1]
            or S.shape[0] != m.shape[0]):
        return None
    det_cov = np.linalg.det(S)
    if det_cov <= 0:
        return None
    inv_cov = np.linalg.inv(S)
    _, d = X.shape
    dif = X - m
    exponent = -0.5 * np.sum(dif @ inv_cov * dif, axis=1)
    norm = 1.0 / (np.sqrt((2 * np.pi) ** d * det_cov))
    pdf_values = norm * np.exp(exponent)
    return np.maximum(pdf_values, 1e-300)
