def evaluate(self, X, Y):
    """
    Evaluates the neuron's predictions

    Args:
        X (numpy.ndarray): Input data of shape (nx, m)
            where nx is the number of input features and m is the number of examples
        Y (numpy.ndarray): Correct labels of shape (1, m)

    Returns:
        numpy.ndarray: The predicted labels for each example of shape (1, m)
        float: The cost of the network for the input data
    """
    # Call the forward propagation method to obtain the neuron's predictions for the input data
    self.forward_prop(X)
    # Calculate the cost of the predictions using the correct labels and the neuron's predicted labels
    cost = self.cost(Y, self.__A)
    # Generate the predicted labels by checking if the neuron's output for each example is greater than or equal to 0.5
    predicted_labels = np.where(self.__A >= 0.5, 1, 0)
    # Return the predicted labels and the cost of the network for the input data
    return predicted_labels, cost
