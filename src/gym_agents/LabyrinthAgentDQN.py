"""
Deep Q-Learning (DQN) agent for labyrinth OpenAI gym environment.
Main to train the Agent.

@authors: Sandra Lassahn, Marc Hensel
@contact: http://www.haw-hamburg.de/marc-hensel
@copyright: 2024
@version: 2024.05.15
@license: CC BY-NC-SA 4.0, see https://creativecommons.org/licenses/by-nc-sa/4.0/deed.en
"""
# conda install pytorch::pytorch danach bspw. in pycharm pip install torch
# conda install matplotlib

import random
import numpy as np
import torch
import torch.optim as optim
import matplotlib.pyplot as plt
from LabyrinthReplayBuffer import LabyrinthReplayBuffer
from LabyrinthQNet import LabyrinthQNet

#Path to access LabyrinthEnvironments
import sys
import os
project_dir = os.path.dirname(os.path.abspath(__file__))
gym_dir = os.path.join(project_dir, '../gym')
sys.path.append(gym_dir)
from LabyrinthEnvironment import LabyrinthEnvironment
project_dir = os.path.dirname(os.path.abspath(__file__))
prototype_dir = os.path.join(project_dir, '../device/Python')
sys.path.append(prototype_dir)
from LabyrinthMachine import LabyrinthMachine

path = "C:/Users/San/Documents/" #lokal Path to load and store weight data

class LabyrinthAgentDQN:

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
            learn_period = 1
    ):
        """
            Constructor.

            Parameters
            ----------
            state_size: int
                size of environment observation space
            action_size: int
                size of action space
            epsilon: float
                exploration rate ε - initial Epsilon greedy parameter,  1 = explore 100% of the time at the beginning
            epsilon_decay_rate: float
                Rate that reduce chance of random action, decay rate of epsilon greedy policy
            epsilon_min: float
                describes how low the exploration rate ε can drop, minimal epsilon value
            batch_size: int
                batch-size for the training
            replay_buffer_size: int
                size of the replay buffer
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
        self.__state_size = state_size #Sets the state size
        self.__action_size = action_size #Sets the action size

        self.__epsilon = epsilon #Sets the initial current epsilon value.
        self.__epsilon_decay_rate = epsilon_decay_rate # Sets the decay rate for epsilon.
        self.__epsilon_min = epsilon_min # Sets the minimum epsilon value.

        # Q-Network initialized in init_q_net method
        self.__learn_period = learn_period #Sets the training frequency
        self.__learning_rate = learning_rate
        self.__init_q_net() # Calls init_q_net to initialize the Q-network.

        # Replay memory
        self.__memory = LabyrinthReplayBuffer(replay_buffer_size, batch_size) #Initializes the replay buffer with the specified size and batch size by calling the init method of the LabyrinthReplayBuffer class.

        self.__training_steps_count = 0 # Initializes the counter for training steps.
        self.__gamma = gamma # Sets the discount factor

        self.__error_print_iteration = 0 # Counter for printing the learning error

    def __init_q_net(self):
        """
            Initialization of the Q-network, optimizer, and loss function

            Parameters
            ----------
            None.

            Returns
            -------
            None.

        """
        self.__q_net = LabyrinthQNet(self.__state_size, self.__action_size) #Initializes the Q-network; LabyrinthQNet is the class of the neural network.
        self.__optimizer = optim.Adam(self.__q_net.parameters(), lr = self.__learning_rate) #optimizer: Adam
        self.__loss = torch.nn.MSELoss() # loss function: Mean Squared Error (MSE)

    def before_episode(self):
        """
            Adjustment of epsilon before a new episode.

            Parameters
            ----------
            None


            Returns
            -------
            None.

        """
        self.__epsilon *= self.__epsilon_decay_rate #Reduces the epsilon value according to the decay rate.
        self.__epsilon = max(self.__epsilon, self.__epsilon_min) #Ensures that epsilon does not fall below the minimum value.

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


    def save(self, path):
        """
            saves the weights of the model in a file.

            Parameters
            ----------
            path: String
                path and name of the file in which it has to be stored

            Returns
            -------
            None
        """
        torch.save(self.__q_net.state_dict(), path)

    def load(self, path):
        """
            Loads the weights of the model from a file.

            Parameters
            ----------
            path: String
                path and name of the file from which it has to be loaded

            Returns
            -------
            None
        """
        self.__q_net.load_state_dict(torch.load(path))

# -----------------------------------------------------------------------------
# Main (sample)
# -----------------------------------------------------------------------------

if __name__ == '__main__':
    # Sets the sequence of random numbers to a defined seed.
    seed = 1
    random.seed(seed)
    np.random.seed(seed)  # Seed for NumPy
    torch.manual_seed(seed)  # Seed for PyTorch (CPU)

    # Init environment and agent
    #env = LabyrinthEnvironment(layout='0 holes real', render_mode='3D') #train with rendered simulation
    env = LabyrinthEnvironment(layout='8 holes', render_mode=None) #training simulation
    #env = LabyrinthMachine(layout='0 holes real', cameraID=0) # training device
    agent = LabyrinthAgentDQN(state_size = 6, action_size = env.num_actions_per_component * 2)
    #save_path = path + '0holesreal.pth' #Uncomment for further training
    #agent.load(save_path) #Uncomment for further training
    episodes = 5500
    scores = []
    # Train agent
    for e in range(1, episodes + 1):
        state, _ = env.reset()
        score = 0
        agent.before_episode()

        while True:
            action = agent.act(state)
            next_state, reward, done, truncated, _ = env.step(action)
            agent.step(state, action, reward, next_state, done)
            state = next_state
            score += reward
            if done or truncated: # or score > 2000 #only for 0 holes
                break

        print(f'Episode {e} Score: {score}')
        scores.append(score)  # save most recent score
        if e % 10 == 0:
            print(f'Episode {e} Average Score: {np.mean(scores[-100:])}')
        if e % 25 == 0: #Saves the weights to a different file every 25 episodes.
            save_path = path + str(e) + '8holes.pth'
            agent.save(save_path)

    # Training results scores
    plt.plot(np.arange(len(scores)), scores)
    plt.ylabel('Score')
    plt.xlabel('Episode')
    plt.show()
