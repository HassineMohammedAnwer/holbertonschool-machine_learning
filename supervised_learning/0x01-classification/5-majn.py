def gradient_descent(self, X, Y, A, alpha=0.05):
    """Calculates one pass of gradient descent on the neuron"""

    # Get the number of examples
    m = Y.shape[1]

    # Calculate the derivative of the cost with respect to z
    dz = A - Y

    # Calculate the derivative of the cost with respect to each weight
    dw = np.matmul(dz, X.T) / m

    # Calculate the derivative of the cost with respect to the bias
    db = np.sum(dz) / m

    # Update the weights using gradient descent
    self.__W = self.__W - (alpha * dw)

    # Update the bias using gradient descent
    self.__b = self.__b - (alpha * db)

