"""
Deep Q-Learning (DQN) agent for labyrinth OpenAI gym environment.

@authors: Sandra Lassahn, Marc Hensel
@contact: http://www.haw-hamburg.de/marc-hensel
@copyright: 2024
@version: 2024.08.29
@license: CC BY-NC-SA 4.0, see https://creativecommons.org/licenses/by-nc-sa/4.0/deed.en
"""
import random
import numpy as np
import torch
import torch.optim as optim
from ReplayBufferDQN import ReplayBuffer
from NetworkDQN import QNet

# TODO Replace by Tensorflow/Keras as standard lib for all projects?

class AgentDQN:

    # ========== Constructor ==================================================

    def __init__(
            self,
            state_size,
            action_size,
            epsilon = 1.0,
            epsilon_decay_rate = .9, #for 0 holes .9, for 2 or 8 holes .98
            epsilon_min = .1, #for 0 holes .1, for 2 or 8 holes 0.15
            batch_size = 64,
            replay_buffer_size = 100_000,
            gamma = 0.99,
            learning_rate = 5e-4,
            learn_period = 1):
        """
        Constructor.

        Parameters
        ----------
        state_size: int
            Size of environment's observation space
        action_size: int
            Size of environment's action space
        epsilon: float
            Exploration rate ε - initial Epsilon greedy parameter,  1 = explore 100% of the time at the beginning
        epsilon_decay_rate: float
            Rate that reduce chance of random action, decay rate of epsilon greedy policy
        epsilon_min: float
            Describes how low the exploration rate ε can drop, minimal epsilon value
        batch_size: int
            Batch size for the training
        replay_buffer_size: int
            Size of the replay buffer
        gamma: float
            Discount factor for past rewards. gamma -> 1: long-term rewards should be given more consideration. gamma -> 0: short-term, immediate optimization of the reward.
        learning_rate: float
            Learning rate 'alpha' for the optimizer
        learn_period: int
            Indicates how often the Q-network should be trained (e.g., 1 = after each executed action)

        Returns
        -------
        None.

        """
        # Environment
        self.__action_size = action_size

        # Epsilon greedy strategy
        self.__epsilon = epsilon
        self.__epsilon_decay_rate = epsilon_decay_rate
        self.__epsilon_min = epsilon_min

        # Q-Network
        self.__learn_period = learn_period      # Training frequency (number of actions after which to train the network)
        self.__learning_rate = learning_rate
        self.__init_q_net(state_size=state_size, action_size=action_size)   # Initialize Q-network

        # Replay memory and further training attributes
        self.__memory = ReplayBuffer(replay_buffer_size, batch_size)
        self.__training_steps_count = 0     # Counter for training steps
        self.__gamma = gamma                # Reinforcement learning discount factor

        # Console output
        self.__print_loss_counter = 0    # Counter for printing the learning error

    # ========== Q-network ====================================================

    def __init_q_net(self, state_size, action_size):
        """
        Initialization of the Q-network, optimizer, and loss function.

        Parameters
        ----------
        state_size: int
            Size of environment's observation space
        action_size: int
            Size of environment's action space

        Returns
        -------
        None.

        """
        self.__q_net = QNet(state_size, action_size)
        self.__optimizer = optim.Adam(self.__q_net.parameters(), lr=self.__learning_rate)
        self.__loss = torch.nn.MSELoss()

    # ========== Adjust epsilon at each new episode ===========================

    def decay_epsilon(self):
        """
        Reduce epsilon for the next episode.
        
        Decays epsilon while ensuring epsilon >= epsilon_min.

        Parameters
        ----------
        None

        Returns
        -------
        None.

        """
        self.__epsilon *= self.__epsilon_decay_rate                 # Reduce epsilon according to the decay rate
        self.__epsilon = max(self.__epsilon, self.__epsilon_min)    # Ensures epsilon does not fall below the minimum value

    # ========== Select next action (epsilon-greedy) ==========================

    def select_action(self, state, mode='train'):
        """
        Select an action according to the epsilon-greedy policy.
        
        Random actions are selected with following probabilities:
            - Training: epsilon
            - Evaluation: 0 %
            
        Else the 'best' known action is selected, being the one with highest
        Q-value. The Q-value is approximated by feeding the state into the
        Q-network.

        Parameters
        ----------
        state: numpy.ndarray
        mode: String, optional
            Possible modes: 'train' or 'evaluate'

        Returns
        -------
        action: int
            Selected action

        """
        # Choose random or greedy action? (Some randomness during evaluation yields better results.)
        random_value = random.random()  # Random value in [0, 1]
        is_random = (mode == 'train' and random_value < self.__epsilon)
        
        # Choose action (randomly or greedy)
        if is_random:
            action = random.choice(np.arange(self.__action_size))
        else:
            # Switch network to evaluation mode (e.g., disables batch normalization)
            self.__q_net.eval()

            # Get network response (action) to state
            state = torch.from_numpy(state).float().unsqueeze(0)    # Convert state from NumPy array to PyTorch tensor and adds an additional dimension to transform the 1D state vector into a 2D batch with a single element. This is important because neural networks typically expect batch processing.
            with torch.no_grad():                                   # Disable gradient computation to speed up execution. (No gradients needed for feed-forward predictions.)
                action_values = self.__q_net(state)                 # Calculate Q-values for all possible actions based on given state
            
            # Switch network back to training mode (for future training steps)
            self.__q_net.train()
            
            # Select action with highest Q-value ('best action')
            action = np.argmax(action_values.data.numpy())
            
        return action

    # ========== Add experience and learn =====================================

    def step(self, state, action, reward, next_state, done):
        """
        Stores an experiences and trigger the training/learning process.
        
        Attribute __learn_period controls how many actions are performed
        before triggering the learning process the next time.

        Parameters
        ----------
        state: numpy.ndarray
        action: int
        reward: float
        next_state: numpy.ndarray
        done: boolean

        Returns
        -------
        None.

        """
        # Add experience to replay buffer
        self.__memory.add(state, action, reward, next_state, done)

        # Trigger learning process if enough samples in replay buffer (every k = learn_process actions)
        self.__training_steps_count += 1
        if self.__training_steps_count == self.__learn_period:
            self.__training_steps_count = 0
            # TODO Rather or additionally let memory.get_random_batch() handle situations with too little data in buffer?
            if len(self.__memory) > 3 * self.__memory.batch_size:
                self.__batch_temporal_difference_step()

    # -------------------------------------------------------------------------

    def __batch_temporal_difference_step(self):
        """
        Trains the neural network by one step of temporal differnce learning.
        
        The step uses a random batch from the replay buffer as training data.
        The network is trained to minimize the temporal difference defined as:
            
            TD = r + g * V(s') - Q(s,a)
                               
        with
            s  : state
            a  : action
            s' : next state
            r  : reward
            g  : discount factor (gamma)
            
        If there is no next state s' (episode terminated), it has no value,
        meaning V(s') = 0 which results in:
               
            TD = r - Q(s,a)
                               
        Parameters
        ----------
        None

        Returns
        -------
        None

        """
        # Get and unpack batch of random samples from replay buffer
        samples = self.__memory.get_random_batch()
        states, actions, rewards, next_states, dones = samples

        # Convert NumPy arrays into PyTorch tensors
        states = torch.from_numpy(states).float()
        actions = torch.from_numpy(actions).long()
        rewards = torch.from_numpy(rewards).float()
        next_states = torch.from_numpy(next_states).float()
        dones = torch.from_numpy(dones).float()

        # V(s') = max(Q(s',a')) of array of next states
        v_next_states = self.__q_net(next_states).detach().max(1)[0].unsqueeze(1)   # Disables gradient computation using .detach() and adds an additional dimension.

        # Q(s,a) of state/action pairs
        q_states = self.__q_net(states)                             # Q-values for the states (and all actions)
        q_states_actions = q_states.gather(dim=1, index=actions)    # Extract Q-values of the chosen actions from the Q-values

        # Temporal difference TD = r + gamma * V(s') - Q(s,a).
        # Expression (1 - dones) is 0 for terminal / last steps in an episode => V(s') = 0 => TD = r - Q(s,a).
        td = rewards + (self.__gamma * v_next_states * (1 - dones)) - q_states_actions

        # Compute loss function between TD and target value 0
        loss = self.__loss(td, torch.zeros(td.shape))
        self.__print_loss_counter += 1
        if self.__print_loss_counter == 100:
            self.__print_loss_counter = 0
            print(f'Loss (each 100 training steps): {loss}')
            
        # Train neural network one step (i.e., minimize TD -> 0)
        # Note: Hyperparameter alpha is the learning rate (set while initializing the optimizer)
        self.__optimizer.zero_grad()    # Reset optimizer's gradients to zero. Gradients are accumulated with each call to backward(). If gradients are not reset, they accumulate with each backpropagation step, leading to incorrect updates.
        loss.backward()                 # Calculat gradient by backpropagation
        self.__optimizer.step()         # Updates the network based on calculated gradients

    # ========== File input / output ==========================================

    def save(self, path):
        """
        Save model's weights to a file.

        Parameters
        ----------
        path: String
            Path and name of the file in which to store the weights

        Returns
        -------
        None
        
        """
        torch.save(self.__q_net.state_dict(), path)

    # -------------------------------------------------------------------------

    def load(self, path):
        """
        Load model's weights from a file.

        Parameters
        ----------
        path: String
            Path and name of the file from which to load the weights

        Returns
        -------
        None
            
        """
        self.__q_net.load_state_dict(torch.load(path))
