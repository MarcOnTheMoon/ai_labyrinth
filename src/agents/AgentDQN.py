"""
Deep Q-Learning (DQN) agent for labyrinth OpenAI gym environment.

@authors: Sandra Lassahn, Marc Hensel
@contact: http://www.haw-hamburg.de/marc-hensel
@copyright: 2024
@version: 2024.08.22
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
            Learning rate for the optimizer
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
        self.__learn_period = learn_period      # Training frequency
        self.__learning_rate = learning_rate
        self.__init_q_net(state_size=state_size, action_size=action_size)   # Initialize Q-network

        # Replay memory
        self.__memory = ReplayBuffer(replay_buffer_size, batch_size)

        self.__training_steps_count = 0 # Initializes the counter for training steps.
        self.__gamma = gamma # Sets the discount factor

        self.__error_print_iteration = 0 # Counter for printing the learning error

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

    def before_episode(self):
        """
        Adjust epsilon before a new episode.

        Parameters
        ----------
        None


        Returns
        -------
        None.

        """
        self.__epsilon *= self.__epsilon_decay_rate                 # Reduce epsilon according to the decay rate
        self.__epsilon = max(self.__epsilon, self.__epsilon_min)    # Ensures epsilon does not fall below the minimum value

    # ==========  ====================================================

    def step(self, state, action, reward, next_state, done):
        """
            Stores experiences and triggers the training/learning process

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
        self.__memory.add(state, action, reward, next_state, done) # Save experience in replay memory

        self.__training_steps_count += 1 # Increments the training steps counter

        if self.__training_steps_count % self.__learn_period == 0: # Checks if it is time to learn.
            # If enough samples are available in memory, get random subset and learn
            if len(self.__memory) > 200: #Checks if there are enough samples in the replay buffer.
                self.__learn() #Calls the learning method.

    # ==========  ====================================================

    def act(self, state, mode = 'train'):
        """
            Selection of an action according to the epsilon-greedy policy.

            Parameters
            ----------
            state: numpy.ndarray
            mode: String, optional
                Possible modes: train or test


            Returns
            -------
            action: int
                chose action to rotate the field.

        """
        r = random.random() #Generates a random number between 0 and 1.
        random_action = mode == 'train' and r < self.__epsilon #Checks if a random action should be selected (in training mode and if r is less than epsilon).
        if mode == 'evaluate' and r < 0.00: #some randomness during evaluation -> better results
            random_action = True
        if random_action: #When a random action should be selected:
            # Random Policy
            action = random.choice(np.arange(self.__action_size)) #Choose a random action
        else:
            # Greedy Policy
            state = torch.from_numpy(state).float().unsqueeze(0)  # Converts the state from a NumPy array to a PyTorch tensor and adds an additional dimension to transform the 1D state vector into a 2D batch with a single element. This is important because neural networks typically expect batch processing.
            self.__q_net.eval()  # Sets the network to evaluation mode (e.g., disables batch normalization).
            with torch.no_grad():  # Disables gradient computation to save memory and speed up execution. During action selection, we do not need gradients since we are only using the model to make predictions.
                action_values = self.__q_net(state)  # Calculates the Q-values for all possible actions based on the given state.
            self.__q_net.train()  # Switches the network back to training mode to ensure it is ready for future training steps.
            action = np.argmax(action_values.data.numpy())  # Selects the action with the highest Q-value. This implements a greedy policy, where the best-known action is always chosen.
        return action #Returns the selected action.

    # ==========  ====================================================

    def __learn(self):
        """
            trains the agent

            Parameters
            ----------
            None

            Returns
            -------
            None

        """
        samples = self.__memory.batch() # Samples experiences from the replay buffer.
        s, a, r, s_next, dones = samples # Unpacks the sample into states s, actions a, rewards r, next states s_next, and end states dones.

        # The following lines convert the NumPy arrays into PyTorch tensors:
        s = torch.from_numpy(s).float()
        a = torch.from_numpy(a).long()
        r = torch.from_numpy(r).float()
        s_next = torch.from_numpy(s_next).float()
        dones = torch.from_numpy(dones).float()

        # V(s') = max(Q(s',a'))
        v_s_next = self.__q_net(s_next).detach().max(1)[0].unsqueeze(1) # Calculates the maximum Q-value for the next states s_next without gradient computation (using .detach()) and adds an additional dimension.

        # Q(s,a)
        q_sa_pure = self.__q_net(s) #Calculates the Q-values for the current state s.
        q_sa = q_sa_pure.gather(dim = 1, index = a) # Extracts the Q-values of the chosen actions a from the computed Q-values.

        # TD = r + g * V(s') - Q(s,a) for non-terminal; TD = r - Q(s,a) for terminal
        td = r + (self.__gamma * v_s_next * (1 - dones)) - q_sa # Calculates the Temporal Difference (TD) error according to the Q-learning update rule. self.__gamma is the discount factor. 1-done is used because there is no subsequent state after a terminal state; thus, V(s′) is 0 in such cases, otherwise V(s′) is 1.

        # Compute loss: TD -> 0
        error = self.__loss(td, torch.zeros(td.shape)) # Calculates the loss between the TD error and zero.
        self.__error_print_iteration = (self.__error_print_iteration + 1) % 100 # Prints the error every 100 steps.
        if self.__error_print_iteration == 0:
            print(f'error: {error}')
        self.__optimizer.zero_grad() # Resets the gradients of the optimizers to zero. Gradients are accumulated by default with each call to backward(). If the gradients are not reset to zero, they would accumulate with each new backpropagation step, leading to incorrect updates.
        error.backward() # Backward= backpropagation: Calculates the gradients of the error with respect to the model parameters.
        self.__optimizer.step() # Updates the network parameters based on the calculated gradients.

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
