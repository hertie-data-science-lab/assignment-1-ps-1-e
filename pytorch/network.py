import time

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


class TorchNetwork(nn.Module):
    def __init__(self, sizes, epochs=10, learning_rate=0.01, random_state=1):
        super().__init__()

        self.epochs = epochs
        self.learning_rate = learning_rate
        self.random_state = random_state

        torch.manual_seed(self.random_state)

        # Define layers
        self.linear1 = nn.Linear(sizes[0], sizes[1])
        self.linear2 = nn.Linear(sizes[1], sizes[2])
        self.linear3 = nn.Linear(sizes[2], sizes[3])

        # Define functions
        self.activation_func = torch.sigmoid
        self.loss_func = nn.CrossEntropyLoss()   # better for classification
        self.optimizer = optim.SGD(self.parameters(), lr=learning_rate)

    def _forward_pass(self, x_train):
        """
        Forward propagation through the network.
        """
        out1 = self.activation_func(self.linear1(x_train))
        out2 = self.activation_func(self.linear2(out1))
        out3 = self.linear3(out2)
        out_softmax = F.softmax(out3, dim=1)  # probabilities
        return out_softmax  


    def _backward_pass(self, y_train, output):
        """
        Backward pass: compute loss and gradients.
        """
        loss = self.loss_func(output, y_train)
        loss.backward()  # autograd takes care of computing gradients
        return loss.item()

    def _update_weights(self):
        """
        Update weights using optimizer (SGD here).
        """
        self.optimizer.step()

    def _flatten(self, x):
        return x.view(x.size(0), -1)
    


    def _print_learning_progress(self, start_time, iteration, train_loader, val_loader):
        train_accuracy = self.compute_accuracy(train_loader)
        val_accuracy = self.compute_accuracy(val_loader)
        print(
        f"Epoch: {iteration + 1}, "
        f"Training Time: {time.time() - start_time:.2f}s, "
        f"Learning Rate: {self.optimizer.param_groups[0]['lr']}, "
        f"Training Accuracy: {train_accuracy * 100:.2f}%, "
        f"Validation Accuracy: {val_accuracy * 100:.2f}%"
        )

    def compute_accuracy(self, data_loader):
        correct = 0
        total = 0
        with torch.no_grad():
                for x, y in data_loader:
                    x = self._flatten(x)
                    preds = self.predict(x)  # uses forward pass + argmax
                    correct += (preds == y).sum().item()
                    total += y.size(0)
        return correct / total

    def predict(self, x):
        """
        Make predictions: return index of most likely output class.
        """
        x = self._flatten(x)
        with torch.no_grad():
            logits = self._forward_pass(x)
            preds = torch.argmax(F(logits, dim=1), dim=1)
        return preds

    def fit(self, train_loader, val_loader):
        start_time = time.time()

        for iteration in range(self.epochs):
            epoch_loss = 0
            for x, y in train_loader:
                x = self._flatten(x)
                y = y.long()  # required for CrossEntropyLoss

                # reset gradients
                self.optimizer.zero_grad()

                # forward + backward + update
                output = self._forward_pass(x)
                loss = self._backward_pass(y, output)
                self._update_weights()

                epoch_loss += loss

            self._print_learning_progress(start_time, iteration, train_loader, val_loader)
            self.history = {
                "loss": [],      # training loss per epoch
                "val_loss": [],  # validation loss per epoch
                "acc": [],       # training accuracy per epoch
                "val_acc": []    # validation accuracy per epoch
            }


