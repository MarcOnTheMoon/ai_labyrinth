"""
Deep Q-Network for a DQN agent solving labyrinth environments.

@authors: Sandra Lassahn, Marc Hensel
@contact: http://www.haw-hamburg.de/marc-hensel
@copyright: 2024
@version: 2024.08.30
@license: CC BY-NC-SA 4.0, see https://creativecommons.org/licenses/by-nc-sa/4.0/deed.en
"""
import torch.nn as nn
import torch.nn.functional as F

# TODO Replace by Tensorflow/Keras as standard lib for all projects?
# TODO Use GPU, if available

# TODO Adapt to layers actually used in training (see following comments)
# 0 holes virtual:
#   Geschwindigkeit im Zustandsraum: 128, 128 (wurde ohne seed trainiert)
#   Letzte Position im Zustandsraum: 512, 128
# 0 holes:          512, 128 (fc1, fc2)
# 2 holes virtual:  128, 128 (fc1, fc2)
# 8 holes:          2048, 1024, 256 (fc1, fc2, fc3)

class QNet(nn.Module):

    # ========== Constructor ==================================================

    def __init__(self, state_size, action_size, qnet):
        """
        Constructor initializing the layers of the neural network.
        
        Note: Attributes must be public to load trained models from files.

        Parameters
        ----------
        state_size: int
            Dimension of the environment's observation space (i.e., network inputs)
        action_size: int
            Dimension of the environment's action space (i.e., network output classes)
        qnet: Dict
            number of neurons in each layer of the neural network, max 3 layers
                "fc1": int,
                    First hidden layer size
                "fc2": int, (optional)
                    Second hidden layer size
                "fc3": int (optional)
                    Third hidden layer size

        Returns
        -------
        None.

        """
        # Build neural network
        super(QNet, self).__init__()  # Necessary to properly initialize inheritance
        fc1_units = qnet.get("fc1")
        fc2_units = qnet.get("fc2")
        fc3_units = qnet.get("fc3")
        self.fc1 = nn.Linear(state_size, fc1_units)  # 1st fully connected layer
        self.bn1 = nn.BatchNorm1d(fc1_units)  # Batch normalization
        if fc2_units != None:
            self.fc2 = nn.Linear(fc1_units, fc2_units)  # 2nd fully connected layer
            self.bn2 = nn.BatchNorm1d(fc2_units)
        else:
            self.fc2 = nn.Linear(fc1_units, action_size)
            self.__net_size = 1
        if fc3_units != None:
            self.fc3 = nn.Linear(fc2_units, fc3_units)  # 3rd fully connected layer
            self.bn3 = nn.BatchNorm1d(fc3_units)
            self.fc4 = nn.Linear(fc3_units, action_size)  # 4th fully connected layer
            self.__net_size = 3
        else:
            self.fc3 = nn.Linear(fc2_units, action_size)
            self.__net_size = 2

        # Init weights (using He-initialization with uniform distribution)
        nn.init.kaiming_uniform_(self.fc1.weight)
        nn.init.kaiming_uniform_(self.fc2.weight)
        if fc2_units != None:
            nn.init.kaiming_uniform_(self.fc3.weight)
        if fc3_units != None:
            nn.init.kaiming_uniform_(self.fc4.weight)

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
        self.fc(x): torch.Tensor
            Computed Q-values for the input state.

        """
        # TODO Input values are not normalized. Normalize input values?
        x = F.leaky_relu(self.bn1(self.fc1(state))) # Pass through 1st layer, batch normalization, and activation function
        if self.__net_size > 1:
            x = F.leaky_relu(self.bn2(self.fc2(x)))
        else: return self.fc2(x)
        if self.__net_size > 2:
            x = F.leaky_relu(self.bn3(self.fc3(x)))
            return self.fc4(x)
        else: return self.fc3(x)
