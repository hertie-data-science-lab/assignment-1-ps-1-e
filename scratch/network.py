import time
import numpy as np
import scratch.utils as utils
from scratch.lr_scheduler import cosine_annealing


class Network():
    def __init__(self, sizes, epochs=50, learning_rate=0.01, random_state=1):
        self.sizes = sizes
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.random_state = random_state
        self.activation_func = utils.sigmoid
        self.activation_func_deriv = utils.sigmoid_deriv
        self.output_func = utils.softmax
        self.output_func_deriv = utils.softmax_deriv
        self.cost_func = utils.mse
        self.cost_func_deriv = utils.mse_deriv

        self.params = self._initialize_weights()


    def _initialize_weights(self):
        # number of neurons in each layer
        input_layer = self.sizes[0]
        hidden_layer_1 = self.sizes[1]
        hidden_layer_2 = self.sizes[2]
        output_layer = self.sizes[3]

        # random initialization of weights
        np.random.seed(self.random_state)
        params = {
            'W1': np.random.rand(hidden_layer_1, input_layer) - 0.5,
            'W2': np.random.rand(hidden_layer_2, hidden_layer_1) - 0.5,
            'W3': np.random.rand(output_layer, hidden_layer_2) - 0.5,
        }

        return params


    def _forward_pass(self, x_train):
        '''
        TODO: Implement the forward propagation algorithm.

        The method should return the output of the network.
        '''
  # In the following section, we ensure that x_train is a column vector
        if len(x_train.shape) == 1: #x_train needs to be converted into a column vector
            x_train = x_train.reshape(-1, 1)
        self.x_input = x_train

        # Store intermediate values for backpropagation
        #In the first step we activate x_train with our weights initialised earlier using the sigmoid function as our first pass
        self.z1 = np.dot(self.params['W1'], x_train)
        self.a1 = self.activation_func(self.z1)
        
        #After our first pass, we repeat the process using a new set of weights that we had initialised earlier
        self.z2 = np.dot(self.params['W2'], self.a1)
        self.a2 = self.activation_func(self.z2)
        
        #In our final pass, we run through the soft max activation function to reach our output
        self.z3 = np.dot(self.params['W3'], self.a2)
        self.a3 = self.output_func(self.z3)
        
        return self.a3


#After we complete the forward part, we go back to update the process using backpropagation
    def _backward_pass(self, y_train, output):
        '''
        TODO: Implement the backpropagation algorithm responsible for updating the weights of the neural network.

        The method should return a dictionary of the weight gradients which are used to update the weights in self._update_weights().

        '''

        # Once again we ensure that y_train is a column vector
        if len(y_train.shape) == 1:
            y_train = y_train.reshape(-1, 1)
    
        m = 1  # Single sample
    
        # Layer 3 (Output): How wrong was the final prediction? 
        # Starting with the output we apply the cost function derivative on y_train and we compute the gradient. This is done to see how much the cost function changes
        dz3 = self.cost_func_deriv(y_train, output)
        dW3 = (1/m) * np.dot(dz3, self.a2.T)  # Gradient for W3
    
        # Layer 2 (Hidden 2): How did this layer contribute to the error?
        # Furthermore, we pass through step 2 and repeat the process. Similarly, we apply the derivative of the activation function 
        da2 = np.dot(self.params['W3'].T, dz3)  # Error passed back
        dz2 = da2 * self.activation_func_deriv(self.z2)  # Apply derivative
        dW2 = (1/m) * np.dot(dz2, self.a1.T)  # Gradient for W2
    
        # Layer 1 (Hidden 1): How did this layer contribute?
        # The final layer goes back to step number 1 where we once again apply the derivative of the activation function 
        da1 = np.dot(self.params['W2'].T, dz2)  # Error passed back
        dz1 = da1 * self.activation_func_deriv(self.z1)  # Apply derivative  
        dW1 = (1/m) * np.dot(dz1, self.x_input.T)  # Gradient for W1
    
        return {'dW1': dW1, 'dW2': dW2, 'dW3': dW3}
    

# Depending upon the gradient we update the weights and multiply it with the learning rate
    def _update_weights(self, weights_gradient, learning_rate):
        '''
        Update the network weights according to stochastic gradient descent.
        '''
        self.params['W1'] -= learning_rate * weights_gradient['dW1']
        self.params['W2'] -= learning_rate * weights_gradient['dW2']
        self.params['W3'] -= learning_rate * weights_gradient['dW3']   
    
        
        

#Printing the learning progress across epochs, training time, training accuracy and validation accuracy
    def _print_learning_progress(self, start_time, iteration, x_train, y_train, x_val, y_val):
        train_accuracy = self.compute_accuracy(x_train, y_train)
        val_accuracy = self.compute_accuracy(x_val, y_val)
        print(
            f'Epoch: {iteration + 1}, ' \
            f'Training Time: {time.time() - start_time:.2f}s, ' \
            f'Training Accuracy: {train_accuracy * 100:.2f}%, ' \
            f'Validation Accuracy: {val_accuracy * 100:.2f}%'
            )


# In this section we predict across all validation samples to check if the predicted class which is the argmax matches the true class ie y = y_hat
    def compute_accuracy(self, x_val, y_val):
    
        predictions = []
        for x, y in zip(x_val, y_val):
            pred = self.predict(x)
            predictions.append(pred == np.argmax(y))

        return np.mean(predictions)


    def predict(self, x):
        '''
        TODO: Implement the prediction making of the network.
        The method should return the index of the most likeliest output class.
        '''
        output = self._forward_pass(x)
        return np.argmax(output)
      


#With the fit function we pass through all the steps of the neural network that we set up earier
    def fit(self, x_train, y_train, x_val, y_val, cosine_annealing_lr=False):

        start_time = time.time()
        total_iters = self.epochs * len(x_train)   # total number of training steps
        t_global = 0                               # global step counter

        #  For this section, we record the history so we can retrieve it while plotting and comparing across models

        self.history = {
            "loss" : [],
            "val_loss": [],
            "acc": [],
            "val_acc": []

        }
#Here, we have set up the learning rate as our actual learning rate and also our cosine learning scheduler
        for iteration in range(self.epochs):
            for x, y in zip(x_train, y_train):

                if cosine_annealing_lr:
                    lr_init = self.learning_rate
                    lr_final = 0.0001   # floor LR
                    learning_rate = cosine_annealing(lr_init, lr_final, t_global, total_iters)
                else:
                    learning_rate = self.learning_rate

                output = self._forward_pass(x)
                weights_gradient = self._backward_pass(y, output)
                self._update_weights(weights_gradient, learning_rate=learning_rate)

                t_global += 1  # increment global step
        
            train_acc = self.compute_accuracy(x_train, y_train)
            val_acc   = self.compute_accuracy(x_val, y_val)

       
            self.history["acc"].append(train_acc)
            self.history["val_acc"].append(val_acc)
            self._print_learning_progress(start_time, iteration, x_train, y_train, x_val, y_val)
        return self.history
