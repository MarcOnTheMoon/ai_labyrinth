"""
Initializes the Deep Q-Network and computes the Q-values of the network.

@authors: Sandra Lassahn, Marc Hensel
@contact: http://www.haw-hamburg.de/marc-hensel
@copyright: 2024
@version: 2024.05.24
@license: CC BY-NC-SA 4.0, see https://creativecommons.org/licenses/by-nc-sa/4.0/deed.en
"""

import torch.nn as nn
import torch.nn.functional as F


class PtQNet(nn.Module): # PtQNet, die von nn.Module erbt. nn.Module ist die Basisklasse für alle neuronalen Netzwerke in PyTorch

    def __init__(self, state_size, action_size, fc1_units = 1024, fc2_units = 512, fc3_units = 128):
        """
            Constructor. initializes the layers of the neural network

            Parameters
            ----------
            state_size: int
                Dimension of the environment observation space
            action_size: int
                Dimension of action space
            fc1_units: int (optional)
                First hidden layer size
            fc2_units: int (optional)
                Second hidden layer size
            fc3_units: int (optional)
                Third hidden layer size

            Returns
            -------
            None.

        """
        super(PtQNet, self).__init__() #Calls the constructor of the base class (nn.Module). This is necessary to properly initialize inheritance
        self.fc1 = nn.Linear(state_size, fc1_units) #Defines the first fully connected layer (Linear Layer) with state_size input neurons and fc1_units output neurons
        self.bn1 = nn.BatchNorm1d(fc1_units) #Performs batch normalization for a 1-dimensional (1D) tensor. 1D Tensor typically in fully connected layers
        self.fc2 = nn.Linear(fc1_units, fc2_units) # Defines the second fully connected layers with fc1_units input neurons and fc2_units output neurons
        self.bn2 = nn.BatchNorm1d(fc2_units)
        self.fc3 = nn.Linear(fc2_units, fc3_units)
        self.bn3 = nn.BatchNorm1d(fc3_units)
        self.fc4 = nn.Linear(fc3_units, action_size) #Defines the fourth fully connected layer with fc3_units input neurons and action_size output neurons. This layer outputs the Q-values for each possible action.

        # Weight initialization using He-initialization with uniform distribution
        nn.init.kaiming_uniform_(self.fc1.weight)
        nn.init.kaiming_uniform_(self.fc2.weight)
        nn.init.kaiming_uniform_(self.fc3.weight)
        nn.init.kaiming_uniform_(self.fc4.weight)

    def forward(self, state):
        """
            Defines the forward pass through the network. It computes the network's output (Q-values) based on a given input (state)

            Parameters
            ----------
            state:
                environment observation space


            Returns
            -------
                self.fc4(x):
                    Compute Q-values for the input state.

        """
        x = F.leaky_relu(self.bn1(self.fc1(state))) #The input state is passed through the first fully connected layer fc1, then through batch normalization, and then through the leaky ReLU activation function. The result is stored in x.
        x = F.leaky_relu(self.bn2(self.fc2(x))) # The output of the first layer is passed through the second fully connected layers fc2, then through batch normalization, and then again through the leaky ReLU activation function. The result is stored in x.
        x = F.leaky_relu(self.bn3(self.fc3(x)))
        return self.fc4(x) #The output of the third layer is passed through fc4. This layer has no activation function because it outputs the final Q-values for each action.
