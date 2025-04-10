#!/usr/bin/env python3
"""5. Bayesian Optimization - Optimization"""
import numpy as np
from scipy.stats import norm
GP = __import__('2-gp').GaussianProcess


class BayesianOptimization:
    """Performs Bayesian optimization on a noiseless 1D Gaussian process"""
    def __init__(self, f, X_init, Y_init, bounds, ac_samples, l=1,
                 sigma_f=1, xsi=0.01, minimize=True):
        self.f = f
        self.X_s = np.linspace(bounds[0], bounds[1], ac_samples).reshape(-1, 1)
        self.xsi = xsi
        self.minimize = minimize
        self.gp = GP(X_init, Y_init, l, sigma_f)

    def acquisition(self):
        """calculates the next best sample location:
        Uses the Expected Improvement acquisition function
        Returns: X_next, EI
        X_next is a numpy.ndarray of shape (1,) representing
        __the next best sample point
        EI is a numpy.ndarray of shape (ac_samples,) containing
        __the expected improvement of each potential sample"""
        mu, sigma = self.gp.predict(self.X_s)
        if self.minimize:
            Y_current = np.min(self.gp.Y)
            improvement = Y_current - mu - self.xsi
        else:
            Y_current = np.max(self.gp.Y)
            improvement = mu - Y_current - self.xsi
        Z = improvement / sigma
        EI = improvement * norm.cdf(Z) + sigma * norm.pdf(Z)
        EI[sigma == 0] = 0
        X_next = self.X_s[np.argmax(EI)]
        return X_next, EI

    def optimize(self, iterations=100):
        """
        optimizes the black-box function:
        iterations is the maximum number of
        __iterations to perform
        If the next proposed point is one that
        __has already been sampled, optimization should be stopped early
        Returns: X_opt, Y_opt
        X_opt is a numpy.ndarray of shape (1,)
        __representing the optimal point
        Y_opt is a numpy.ndarray of shape (1,)
        __representing the optimal function value
        """
        for _ in range(iterations):
            X_next, _ = self.acquisition()
            old_X_next = X_next
            X_next, _ = self.acquisition()
            Y_next = self.f(X_next)
            if X_next == old_X_next:
                break

            self.gp.update(X_next, Y_next)
        if self.minimize:
            idx = np.argmin(self.gp.Y)
        else:
            idx = np.argmax(self.gp.Y)
        X_opt = self.gp.X[idx]
        Y_opt = self.gp.Y[idx]
        return X_opt, Y_opt
