# Brady Young
# This file contains the node definitions for both the output
# and hidden layers. Each calculates their errors and updates their weights
# in slightly different ways in my implementation so they each have their own class.

import numpy as np
import Activations as activate


# Class to hold the data members of subclasses
class Node:
    def __init__(self, outputs, inputs):
        self.weights = np.random.rand(outputs, inputs)
        self.weights -= 0.5  # Scales the weights
        self.activated = []  # List to hold activated values for each node
        self.error = None  # Holds the error value for each node


# Defines the node functions for the Hidden Layer #
class Hidden(Node):
    # Creates a list of each hidden node's activated values
    def activation(self, array, activation):
        self.activated = []

        if activation is "sig":
            for a in array:
                self.activated.append(activate.sigmoid(a))
        else:
            for a in array:
                self.activated.append(activate.arctan(a))

        # attach bias
        self.activated.append(1)
        # convert activated list to numpy array
        self.activated = np.array(self.activated)

    #  -Calculates each nodes error based on output node error and current activations
    #  -Sets the error for each node
    #  -Removes the bias from activated list and error list
    #    The bias is included up to the error calculation,
    #    but is removed afterwards because the bias node value
    #    is static
    def calcError(self, output, activation):
        if activation is "sig":
            term_two = (1 - self.activated)
            term_three = output.weights * output.error.reshape((-1, 1))
            error = self.activated * term_two * np.sum(term_three, axis=0)

            # Error is set to none after updating weights
            if self.error is None:
                self.error = error
            else:
                self.error += error

        # Arctan derivative
        else:
            term_one = output.weights * output.error.reshape((-1, 1))
            term_one = np.sum(term_one, axis=0)
            error = 1 / (1 + term_one)

            if self.error is None:
                self.error = error
            else:
                self.error += error

    # Updates the weight values based on the error and learning rate
    def update(self, features, params):
        self.error = np.delete(self.error, params.num_hidden)  # bias removed
        change = params.rate * np.outer(self.error, features)
        self.weights += (change / params.batch_size)

        # Clear the error for next batch
        self.error = None


# Defines the node functions for the output layer
class Output(Node):
    # Creates a list of each output node's activated values
    def activation(self, array, activation):
        self.activated = []
        if activation is "sig":
            for a in array:
                self.activated.append(activate.sigmoid(a))

        else:
            for a in array:
                self.activated.append(activate.arctan(a))

        self.activated = np.array(self.activated)

    # Calculates error for output nodes
    def calcError(self, target, activation):
        if activation is "sig":
            term_two = 1 - self.activated
            term_three = target - self.activated
            error = self.activated * term_two * term_three

            if self.error is None:
                self.error = error
            else:
                self.error += error

        else:
            term_one = target - self.activated
            error = 1 / (1 + term_one)

            if self.error is None:
                self.error = error
            else:
                self.error += error

    # Updates the output node weights
    def update(self, inputs, params):
        if hasattr(inputs, 'activated'):
            change = params.rate * np.outer(self.error, inputs.activated)
        else:
            change = params.rate * np.outer(self.error, inputs)

        self.weights += (change / params.batch_size)
        self.error = None
