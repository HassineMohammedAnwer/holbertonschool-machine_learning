#!/usr/bin/env python3
"""1. Gaussian Process Prediction"""
import numpy as np


class GaussianProcess:
    """represents a noiseless 1D Gaussian process"""
    def __init__(self, X_init, Y_init, l=1, sigma_f=1):
        """X_init is a numpy.ndarray of shape (t, 1) representing
        __the inputs already sampled with the black-box function
        Y_init is a numpy.ndarray of shape (t, 1) representing the
        __outputs of the black-box function for each input in X_init
        t is the number of initial samples
        l is the length parameter for the kernel
        sigma_f is the standard deviation given to the output of the
        __black-box function
        Sets the public instance attributes X, Y, l, and sigma_f
        __corresponding to the respective constructor inputs
        Sets the public instance attribute K, representing the current
        __covariance kernel matrix for the Gaussian process"""
        self.X = X_init
        self.Y = Y_init
        self.l = l
        self.sigma_f = sigma_f
        self.K = self.kernel(self.X, self.X)

    def kernel(self, X1, X2):
        """calculates the covariance kernel matrix between two matrices:
        X1 is a numpy.ndarray of shape (m, 1)
        X2 is a numpy.ndarray of shape (n, 1)
        the kernel should use the Radial Basis Function (RBF)
        sigma_f ** 2 *
           exp(-1/2 * squared Euclidean distance between the points / l )
        Returns: the covariance kernel matrix as a numpy.ndarray(m, n)
        """
        sqdist = np.sum(X1**2, 1).reshape(-1,
                                          1) + np.sum(X2**2,
                                                      1) - 2 * np.dot(X1, X2.T)
        K = self.sigma_f**2 * np.exp(-0.5 * sqdist / self.l**2)
        return K

    def predict(self, X_s):
        """ predicts the mean and standard deviation of points in a
        Gaussian process:
        X_s is a numpy.ndarray of shape (s, 1) containing all of the
        points whose mean and standard deviation should be calculated
        s is the number of sample points
        Returns: mu, sigma
        mu is a numpy.ndarray of shape (s,) containing the mean for
        each point in X_s, respectively
        sigma is a numpy.ndarray of shape (s,) containing the variance
        for each point in X_s, respectively"""
        K_s = self.kernel(self.X, X_s)
        K_m = self.kernel(X_s, X_s)
        K_x = np.linalg.inv(self.K)
        
        mu = (K_s.T.dot(K_x).dot(self.Y)).reshape(-1)
        sigma = np.diag(K_m - K_s.T.dot(K_x).dot(K_s))
        
        return mu, sigma
