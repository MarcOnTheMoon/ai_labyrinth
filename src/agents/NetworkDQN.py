"""
Deep Q-Network for a DQN agent solving labyrinth environments.

@authors: Sandra Lassahn, Marc Hensel
@contact: http://www.haw-hamburg.de/marc-hensel
@copyright: 2024
@version: 2024.08.26
@license: CC BY-NC-SA 4.0, see https://creativecommons.org/licenses/by-nc-sa/4.0/deed.en
"""
import torch.nn as nn
import torch.nn.functional as F

# TODO Replace by Tensorflow/Keras as standard lib for all projects?
# TODO Use GPU, if available

class QNet(nn.Module):

    # ========== Constructor ==================================================

    def __init__(self, state_size, action_size, fc1_units=4096, fc2_units=2048, fc3_units=512):
        """
        Constructor initializing the layers of the neural network.

        Parameters
        ----------
        state_size: int
            Dimension of the environment's observation space (i.e., network inputs)
        action_size: int
            Dimension of the environment's action space (i.e., network output classes)
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
        # Build neural network
        super(QNet, self).__init__()                    # Necessary to properly initialize inheritance
        self.__fc1 = nn.Linear(state_size, fc1_units)     # 1st fully connected layer
        self.__bn1 = nn.BatchNorm1d(fc1_units)            # Batch normalization
        self.__fc2 = nn.Linear(fc1_units, fc2_units)      # 2nd fully connected layer
        self.__bn2 = nn.BatchNorm1d(fc2_units)
        self.__fc3 = nn.Linear(fc2_units, fc3_units)      # 3rd fully connected layer
        self.__bn3 = nn.BatchNorm1d(fc3_units)
        self.__fc4 = nn.Linear(fc3_units, action_size)    # 4th fully connected layer

        # Init weights (using He-initialization with uniform distribution)
        nn.init.kaiming_uniform_(self.__fc1.weight)
        nn.init.kaiming_uniform_(self.__fc2.weight)
        nn.init.kaiming_uniform_(self.__fc3.weight)
        nn.init.kaiming_uniform_(self.__fc4.weight)

    # ========== Get network output for specific input (feed-forward) =========

    def forward(self, state):
        """
        Forward pass through the network.
        
        The method computes the network's output (Q-values) based on a given
        input (state).

        Parameters
        ----------
        state: numpy.ndarray
            environment observation space

        Returns
        -------
        self.__fc4(x): torch.Tensor
            Computed Q-values for the input state.

        """
        # TODO Are input values not normalized?! Normalize input values?
        x = F.leaky_relu(self.__bn1(self.__fc1(state))) # Pass through 1st layer, batch normalization, and activation function
        x = F.leaky_relu(self.__bn2(self.__fc2(x)))
        x = F.leaky_relu(self.__bn3(self.__fc3(x)))
        return self.__fc4(x)
