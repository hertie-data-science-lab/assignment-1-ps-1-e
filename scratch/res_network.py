import time
import numpy as np
from scratch.network import Network

class ResNetwork(Network):

    def __init__(self, sizes, epochs=50, learning_rate=0.01, random_state=1):
        """
        Residual Network inheriting from Network.
        Residual connections are applied only when input/output dimensions match.
        """
        super().__init__(sizes, epochs, learning_rate, random_state)

    def _forward_pass(self, x_train):
        """
        Forward propagation with conditional residual connections.
        """
        if len(x_train.shape) == 1:
            x_train = x_train.reshape(-1, 1)
        self.x_input = x_train

        # Hidden layer 1
        self.z1 = np.dot(self.params['W1'], self.x_input)
        h1 = self.activation_func(self.z1)

        if h1.shape == self.x_input.shape:
            self.a1 = h1 + self.x_input
            self.res1_used = True
            print(" Residual connection applied at Layer 1")
        else:
            self.a1 = h1
            self.res1_used = False
            print("⚠️ Residual connection skipped at Layer 1 (shape mismatch)")

        # Hidden layer 2
        self.z2 = np.dot(self.params['W2'], self.a1)
        h2 = self.activation_func(self.z2)

        if h2.shape == self.a1.shape:
            self.a2 = h2 + self.a1
            self.res2_used = True
            print(" Residual connection applied at Layer 2")
        else:
            self.a2 = h2
            self.res2_used = False
            print("⚠️ Residual connection skipped at Layer 2 (shape mismatch)")

        # Output layer
        self.z3 = np.dot(self.params['W3'], self.a2)
        self.a3 = self.output_func(self.z3)

        return self.a3

    def _backward_pass(self, y_train, output):
        """
        Backpropagation with conditional residual connections.
        """
        if len(y_train.shape) == 1:
            y_train = y_train.reshape(-1, 1)
        m = 1

        # Output layer
        dz3 = self.cost_func_deriv(y_train, output)
        dW3 = (1/m) * np.dot(dz3, self.a2.T)

        # Hidden layer 2
        da2 = np.dot(self.params['W3'].T, dz3)
        dz2 = da2 * self.activation_func_deriv(self.z2)
        dW2 = (1/m) * np.dot(dz2, self.a1.T)

        # Add gradient from residual path if used
        da1_from_residual = da2 if self.res2_used else 0

        # Hidden layer 1
        da1 = np.dot(self.params['W2'].T, dz2) + da1_from_residual
        dz1 = da1 * self.activation_func_deriv(self.z1)
        dW1 = (1/m) * np.dot(dz1, self.x_input.T)

        return {'dW1': dW1, 'dW2': dW2, 'dW3': dW3}
