#!/usr/bin/env python3
""" 28. All the Activations """
import numpy as np
import pickle
import os


class DeepNeuralNetwork:
    """DeepNeuralNetwork class"""
    def __init__(self, nx, layers, activation='sig'):
        """DeepNeuralNetwork class constructor
        Add an activation parameter to the class constructor,
        __validate it, and store it as a private attribute."""
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if not isinstance(layers, list) or len(layers) == 0:
            raise TypeError("layers must be a list of positive integers")
        if activation not in ['sig', 'tanh']:
            raise ValueError("activation must be 'sig' or 'tanh'")

        self.__activation = activation
        self.__L = len(layers)
        self.__cache = {}
        self.__weights = {'W1':
                          np.random.randn(layers[0], nx) * np.sqrt(2 / nx),
                          'b1': np.zeros((layers[0], 1))
                          }
        if not isinstance(layers[0], int) or layers[0] <= 0:
            raise TypeError("layers must be a list of positive integers")

        for i in range(1, self.__L):
            if not isinstance(layers[i], int) or layers[i] <= 0:
                raise TypeError("layers must be a list of positive integers")
            self.__weights["W" + str(i + 1)] =\
                np.random.randn(layers[i], layers[i - 1]) *\
                np.sqrt(2 / layers[i - 1])
            self.__weights["b" + str(i + 1)] = np.zeros((layers[i], 1))

    @property
    def L(self):
        """ number of layers in the neural network"""
        return self.__L

    @property
    def weights(self):
        """ A dictionary to hold all intermediary values of the network """
        return self.__weights

    @property
    def cache(self):
        """A dictionary to hold all weights and biased of the network """
        return self.__cache

    @property
    def activation(self):
        """Activation function getter"""
        return self.__activation

    def forward_prop(self, X):
        """forward propagation of the neural network
        change:  use softmax activation in the output layer
        __for multiclass classification,
        __while retaining sigmoid activation for hidden layers.
        Use the specified activation function (sigmoid or tanh)
        __for hidden layers and softmax for the output layer."""
        self.__cache["A0"] = X
        A_prev = X
        for i in range(1, self.L + 1):
            Z = np.dot(self.weights[f"W{i}"], A_prev) + self.weights[f"b{i}"]
            if i == self.L:
                exp_Z = np.exp(Z - np.max(Z, axis=0, keepdims=True))
                A = exp_Z / np.sum(exp_Z, axis=0, keepdims=True)
            else:
                if self.__activation == 'sig':
                    A = 1 / (1 + np.exp(-Z))
                else:
                    A = np.tanh(Z)
            self.__cache[f"A{i}"] = A
            A_prev = A
        return A, self.cache

    def cost(self, Y, A):
        """cost using Cross-entropy"""
        m = Y.shape[1]
        A_clipped = np.clip(A, 1e-8, 1 - 1e-8)
        return (-1 / m) * np.sum(Y * np.log(A_clipped))

    def evaluate(self, X, Y):
        """Return: Returns the neuron’s prediction and the cost
           prediction: a numpy.ndarray with shape (1, m) of predicted labels
           for each example predicted labels for each example and the label
           values should be 1
           The label values should be 1 if the output of the network is >= 0.5
           and 0 otherwise
         converts the softmax output to one-hot encoded predictions by taking
         __argmax of the output probabilities for each example, ensuring the
         __predictions match the format of the input labels."""
        A, _ = self.forward_prop(X)
        predictions = np.where(A >= 0.5, 1, 0)
        classes = A.shape[0]
        one_hot_predictions = np.eye(classes)[predictions].T
        cost = self.cost(Y, A)
        return predictions, cost

    def gradient_descent(self, Y, cache, alpha=0.05):
        """method calculates one pass of gradient descent on the neuron
        # Number of examples
        m = Y.shape[1]
        # Derivative of the loss with respect to Z_L
        dz = A - Y
        dw = np.matmul(dz, X.T) / m
        db = np.sum(dz) / m
        # Update the weights and biases
        self.__W = self.__W - (alpha * dw)
        self.__b = self.__b - (alpha * db)
        Compute the derivatives of the activation function during
        __backpropagation to update the weights correctly."""
        m = Y.shape[1]
        L = self.L
        A = cache[f"A{L}"]
        dZ = A - Y

        for layer in reversed(range(1, L + 1)):
            A_prev = cache[f"A{layer - 1}"]
            W = self.weights[f"W{layer}"]
            dW = (1 / m) * np.dot(dZ, A_prev.T)
            db = (1 / m) * np.sum(dZ, axis=1, keepdims=True)

            if layer > 1:
                if self.__activation == 'sig':
                    deriv = A_prev * (1 - A_prev)
                else:
                    deriv = 1 - (A_prev ** 2)
                dZ = np.dot(W.T, dZ) * deriv

            self.weights[f"W{layer}"] -= alpha * dW
            self.weights[f"b{layer}"] -= alpha * db

    def train(self, X, Y, iterations=5000, alpha=0.05,
              verbose=True, graph=True, step=100):
        """for i in range(iterations):
            A1, A2 = self.forward_prop(X)
            self.gradient_descent(X, Y, A1, A2, alpha)
        Verbose Output: Print the cost at every specified interval
        __if verbose mode is enabled.
        Graph Plotting: Plot the training cost over iterations
        __using matplotlib if graph mode is enabled.
        A, cost = self.evaluate(X, Y)
        return A, cost"""
        if not isinstance(iterations, int):
            raise TypeError("iterations must be an integer")
        if iterations < 0:
            raise ValueError("iterations must be a positive integer")
        if not isinstance(alpha, float):
            raise TypeError("alpha must be a float")
        if alpha < 0:
            raise ValueError("alpha must be positive")
        if verbose or graph:
            if not isinstance(step, int):
                raise TypeError("step must be an integer")
            if step <= 0 or step > iterations:
                raise ValueError("step must be positive and <= iterations")
        steps_to_log = []
        costs = []
        if verbose or graph:
            steps_to_log = list(range(0, iterations + 1, step))
            if not steps_to_log or steps_to_log[-1] != iterations:
                steps_to_log.append(iterations)
            # Compute initial cost
            A0, _ = self.forward_prop(X)
            cost0 = self.cost(Y, A0)
            costs.append(cost0)
            if verbose:
                print(f"Cost after 0 iterations: {cost0}")
        i = 0
        while (i < iterations):
            # run forward propagation
            self.forward_prop(X)
            # run gradient descent
            self.gradient_descent(Y, self.cache, alpha)
            current_i = i + 1

            if verbose or graph:
                if current_i in steps_to_log:
                    A_new, _ = self.forward_prop(X)
                    cost = self.cost(Y, A_new)
                    costs.append(cost)
                    if verbose:
                        print(f"Cost after {current_i} iterations: {cost}")
            i += 1

        if graph and (verbose or graph):
            plt.plot(steps_to_log, costs, 'b-')
            plt.xlabel('iteration')
            plt.ylabel('cost')
            plt.title('Training Cost')
            plt.show()

        return self.evaluate(X, Y)

    def save(self, filename):
        """Saves the instance object to a file in pickle format
        filename is the file to which the object should be saved
        If filename does not have the extension .pkl, add it"""
        if not filename.endswith('.pkl'):
            filename += '.pkl'
        with open(filename, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load(filename):
        """Loads a pickled DeepNeuralNetwork object
        filename is the file from which the object should be loaded
        Returns: the loaded object, or None if filename doesn’t exist"""
        if not os.path.exists(filename):
            return None
        with open(filename, 'rb') as f:
            return pickle.load(f)
