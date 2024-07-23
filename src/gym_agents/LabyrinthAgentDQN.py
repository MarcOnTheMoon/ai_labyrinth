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
from replay_buffer import ReplayBuffer
from q_model import PtQNet

#Path to access LabyrinthEnvironment
import sys
import os
project_dir = os.path.dirname(os.path.abspath(__file__))
gym_dir = os.path.join(project_dir, '../gym')
sys.path.append(gym_dir)
from LabyrinthEnvironment import LabyrinthEnvironment
project_dir = os.path.dirname(os.path.abspath(__file__))
prototype_dir = os.path.join(project_dir, '../prototype')
sys.path.append(prototype_dir)
from LabyrinthEnvironmentPrototype import LabyrinthEnvironmentPrototype

path = "C:/Users/Sandra/Documents/" #lokal Path to load and store weight data

class DqnAgent:

    def __init__(
            self,
            state_size,
            action_size,
            degp_epsilon = 1,
            degp_decay_rate = .9, #for 0 holes .9, for 2 or 8 holes .98
            degp_min_epsilon = .1, #for 0 holes .1, for 2 or 8 holes 0.15
            train_batch_size = 64,
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
            degp_epsilon: int
                exploration rate ε - initial Epsilon greedy parameter,  1 = explore 100% of the time at the beginning
            degp_decay_rate: float
                Rate that reduce chance of random action, decay rate of epsilon greedy policy
            degp_min_epsilon: float
                describes how low the exploration rate ε can drop, minimal epsilon value
            train_batch_size: int
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
        self.state_size = state_size #Sets the state size
        self.action_size = action_size #Sets the action size

        self.degp_epsilon = self.degp_initial_epsilon = degp_epsilon #Sets the initial and current epsilon value.
        self.degp_decay_rate = degp_decay_rate # Sets the decay rate for epsilon.
        self.degp_min_epsilon = degp_min_epsilon # Sets the minimum epsilon value.

        # Q-Network initialized in init_q_net method
        self.learn_period = learn_period #Sets the training frequency
        self.learning_rate = learning_rate
        self.init_q_net() # Calls init_q_net to initialize the Q-network.

        # Replay memory
        self.memory = ReplayBuffer(replay_buffer_size, train_batch_size) #Initializes the replay buffer with the specified size and batch size by calling the init method of the ReplayBuffer class.

        self.training_steps_count = 0 # Initializes the counter for training steps.
        self.train_batch_size = train_batch_size # Sets the batch size for training.
        self.replay_buffer_size = replay_buffer_size #sets the size of the replay buffer.
        self.gamma = gamma # Sets the discount factor

        self.error_print_iteration = 0 # Counter for printing the learning error

    def init_q_net(self):
        """
            Initialization of the Q-network, optimizer, and loss function

            Parameters
            ----------
            None.

            Returns
            -------
            None.

        """
        self.q_net = PtQNet(self.state_size, self.action_size) #Initializes the Q-network; PtQNet is the class of the neural network.
        self.optimizer = optim.Adam(self.q_net.parameters(), lr = self.learning_rate) #optimizer: Adam
        self.loss = torch.nn.MSELoss() # loss function: Mean Squared Error (MSE)

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
        self.degp_epsilon *= self.degp_decay_rate #Reduces the epsilon value according to the decay rate.
        self.degp_epsilon = max(self.degp_epsilon, self.degp_min_epsilon) #Ensures that epsilon does not fall below the minimum value.

    def step(self, state, action, reward, next_state, done):
        """
            Stores experiences and triggers the training/learning process

            Parameters
            ----------
            state:
            action: int
            reward: float
            next_state:
            done: boolean


            Returns
            -------
            None.

        """
        self.memory.add(state, action, reward, next_state, done) # Save experience in replay memory

        self.training_steps_count += 1 # Increments the training steps counter

        if self.training_steps_count % self.learn_period == 0: # Checks if it is time to learn.
            # If enough samples are available in memory, get random subset and learn
            if len(self.memory) > 200: #Checks if there are enough samples in the replay buffer.
                self.learn() #Calls the learning method.

    def act(self, state, mode = 'train'):
        """
            Selection of an action according to the epsilon-greedy policy.

            Parameters
            ----------
            state: numpy array
            mode: String
                Possible modes: train or test


            Returns
            -------
            action:
                chose action to rotate the field.

        """
        r = random.random() #Generates a random number between 0 and 1.
        random_action = mode == 'train' and r < self.degp_epsilon #Checks if a random action should be selected (in training mode and if r is less than epsilon).
        if mode == 'test' and r < 0.00: #smal randomness during evaluation -> better results
            random_action = True
        if random_action: #When a random action should be selected:
            # Random Policy
            action = random.choice(np.arange(self.action_size)) #Choose a random action
        else:
            # Greedy Policy
            state = torch.from_numpy(state).float().unsqueeze(0)  # Converts the state state from a NumPy array to a PyTorch tensor and adds an additional dimension to transform the 1D state vector into a 2D batch with a single element. This is important because neural networks typically expect batch processing.

            self.q_net.eval()  # Sets the network to evaluation mode (e.g., disables batch normalization).

            with torch.no_grad():  # Disables gradient computation to save memory and speed up execution. During action selection, we do not need gradients since we are only using the model to make predictions.
                action_values = self.q_net(state)  # Calculates the Q-values for all possible actions based on the given state.
            self.q_net.train()  # Switches the network back to training mode to ensure it is ready for future training steps.
            action = np.argmax(action_values.data.numpy())  # Selects the action with the highest Q-value. This implements a greedy policy, where the best-known action is always chosen.

        return action #Returns the selected action.

    def learn(self):
        """
            trains the agent

            Parameters
            ----------
            None

            Returns
            -------
            None

        """
        samples = self.memory.batch() # Samples experiences from the replay buffer.
        s, a, r, s_next, dones = samples # Unpacks the sample into states s, actions a, rewards r, next states s_next, and end states dones.

        # The following lines convert the NumPy arrays into PyTorch tensors:
        s = torch.from_numpy(s).float()
        a = torch.from_numpy(a).long()
        r = torch.from_numpy(r).float()
        s_next = torch.from_numpy(s_next).float()
        dones = torch.from_numpy(dones).float()

        # V(s') = max(Q(s',a))
        v_s_next = self.q_net(s_next).detach().max(1)[0].unsqueeze(1) # Calculates the maximum Q-value for the next states s_next without gradient computation (using .detach()) and adds an additional dimension.

        # Q(s,a)
        q_sa_pure = self.q_net(s) #Calculates the Q-values for the current state s.
        q_sa = q_sa_pure.gather(dim = 1, index = a) # Extracts the Q-values of the chosen actions a from the computed Q-values.

        # TD = r + g * V(s') - Q(s,a)
        td = r + (self.gamma * v_s_next * (1 - dones)) - q_sa # Calculates the Temporal Difference (TD) error according to the Q-learning update rule. self.gamma is the discount factor. 1-done is used because there is no subsequent state after a terminal state; thus, V(s′) is 0 in such cases, otherwise V(s′) is 1.

        # Compute loss: TD -> 0
        error = self.loss(td, torch.zeros(td.shape)) # Calculates the loss between the TD error and zero.
        self.error_print_iteration = (self.error_print_iteration + 1) % 100 # Prints the error every 100 steps.
        if self.error_print_iteration == 0:
            print(f'error: {error}')
        self.optimizer.zero_grad() # Resets the gradients of the optimizers to zero. Gradients are accumulated by default with each call to backward(). If the gradients are not reset to zero, they would accumulate with each new backpropagation step, leading to incorrect updates.
        error.backward() # Backward= backpropagation: Calculates the gradients of the error with respect to the model parameters.
        self.optimizer.step() # Updates the network parameters based on the calculated gradients.


    def save(self, path):
        """
            saves the weights of the model in a file.

            Parameters
            ----------
            path:
                path and name of the file in which it has to be stored


            Returns
            -------
            None
        """
        torch.save(self.q_net.state_dict(), path)

    def load(self, path):
        """
            Loads the weights of the model from a file.

            Parameters
            ----------
            path:
                path and name of the file from which it has to be loaded


            Returns
            -------
            None
        """
        self.q_net.load_state_dict(torch.load(path))

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
    env = LabyrinthEnvironment(layout='2 holes real', render_mode=None) #training simulation
    #env = LabyrinthEnvironmentPrototype(layout='0 holes real', cameraID=0) # training prototype
    agent = DqnAgent(state_size = 6, action_size = env.num_actions_per_component * 2)
    #save_path = path + '0holesreal.pth'
    #agent.load(save_path)
    episodes = 1000
    scores = []
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
            save_path_100 = path + str(e) + '0holesreal_.pth'
            agent.save(save_path_100)

    # Training Results scores
    plt.plot(np.arange(len(scores)), scores)
    plt.ylabel('Score')
    plt.xlabel('Episode')
    plt.show()
