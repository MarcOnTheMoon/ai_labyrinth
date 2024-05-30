"""
Deep Q-Learning (DQN) agent for labyrinth OpenAI gym environment.

@authors: Sandra Lassahn, Marc Hensel
@contact: http://www.haw-hamburg.de/marc-hensel
@copyright: 2024
@version: 2024.05.15
@license: CC BY-NC-SA 4.0, see https://creativecommons.org/licenses/by-nc-sa/4.0/deed.en
"""
#conda install -c conda-forge tensorflow

import random
import numpy as np
from collections import deque
from tensorflow.keras import Sequential, Input
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers import Adam

#Path to access LabyrinthEnvironment
import sys
import os
project_dir = os.path.dirname(os.path.abspath(__file__))
gym_dir = os.path.join(project_dir, '../gym')
sys.path.append(gym_dir)
from LabyrinthEnvironment import LabyrinthEnvironment

path = "C:/Users/Sandra/Documents/" #lokal Path to load and store weight data

class LabyrinthAgentDQN:
    def __init__(self, state_space, action_space):
        """
            Constructor.

            Parameters
            ----------


            Returns
            -------
            None.

        """

        self.state_space = state_space #till now not used
        self.action_space = action_space #till now not used
        self.memory = deque(maxlen=2000) #The memories are stored as elements of a data structure called "deque" ("deck"), which functions similarly to a list. It only retains the most recent maxlength elements.
        self.memory_important_rewards = deque(maxlen=1) #store target rewards
        self.gamma = 0.97 # originally 0.95, Discount factor for past rewards. gamma -> 1: long-term rewards should be given more consideration. gamma -> 0: short-term, immediate optimization of the reward.
        self.epsilon = 1.0 # exploration rate ε - Epsilon greedy parameter,  1.0 = explore 100% of the time at the beginning
        self.exploration_min = 0.1 #originally 0.01, describes how low the exploration rate ε can drop
        self.exploration_decay = 0.99  # originally 0.99, Rate at which to reduce chance of random action being taken as the agent gets better and better in playing
        self.learning_rate = 0.001 # originally 0.001, Learning rate for the optimizer - hyperparameter

        self.model = self._build_model()

    def _build_model(self):
        """
            neural network architecture

            Parameters
            ----------
            None

            Returns
            -------
            model: Sequential
                the neural network

        """
        model = Sequential()
        model.add(Input(shape=(6,), dtype='float32', name='state')) # input layer
        model.add(Dense(32, activation='elu')) # hidden layer: In TensorFlow and Keras, the glorot_uniform initializer is used by default for Dense layers, also known as Xavier initialization.
        model.add(Dense(64, activation='elu')) # Dense is the basic form of a neural network layer
        model.add(Dense(32, activation='elu'))
        model.add(Dense(9*9, activation='linear', name='action')) # output layer
        model.summary() # Displays a summary of the model, including the number of parameters per layer
        model.compile(loss='mse', optimizer=Adam(learning_rate=self.learning_rate))  #mse = mean_squared_error, also possible mae = mean_absolut_error or mean_q = average Q-value
        return model

    def remember(self, state, action, reward, next_state, done):
        """
            simply store states, actions and resulting rewards into the memory.
            Replay Memory is used to store experiences and later use them for training the model.

            Parameters
            ----------
            state:
            action:
            reward:
            next_state:
            done:

            Returns
            -------
            None

        """

        self.memory.append((state, action, reward, next_state, done))
        if reward > 100 or reward < -100: # remembers important experiences (target and holes)
            self.memory_important_rewards.append((state, action, reward, next_state, done))

    def act(self, state):
        """
            uses Epsilon-Greedy policy for action selection

            Parameters
            ----------
            state:


            Returns
            -------
            x_action:
                action to rotate the field in the x direction.
            y_action:
                action to rotate the field in the y direction.
        """
        if np.random.rand() <= self.epsilon:
            # exploration: take random action out of 9 different actions
            x_action = np.random.choice([0,1,2,3,4,5,6,7,8])
            y_action = np.random.choice([0,1,2,3,4,5,6,7,8])
            return [x_action, y_action]
        else:
            # exploitation: take the best action based on the trained data
            state = np.array(state).reshape(1, -1) # Shape the state into the correct format
            q_values = self.model.predict(state) # predict(): Model predicts the reward of the current state based on the data trained so far
            act_values = q_values.reshape((9, 9)) # convert to a 2D array representing the (x, y) action grid
            x_action, y_action = np.unravel_index(np.argmax(act_values), act_values.shape) #Take the best action with converting into the dimension of (x, y) coordinates.
            return [x_action, y_action]

    def train(self, batch_size):
        """
            trains the neural network with samples from the memory.

            Parameters
            ----------
            batch_size: int
                Number of samples to be used for training


            Returns
            -------
            None
        """
        minibatch = random.sample(self.memory, batch_size) #only take a few samples (batch_size) out of self.memory, pick them randomly.
        if len(self.memory_important_rewards) > 0:
            for state, action, reward, next_state, done in self.memory_important_rewards:
                minibatch.append((state, action, reward, next_state, done)) # always learn important experiences (target and hole)

        for state, action, reward, next_state, done in minibatch:
            # format into the correct shape
            state = np.array(state).reshape(1, -1)
            next_state = np.array(next_state).reshape(1, -1)

            # prediction of the Q-values for the current state
            target_f = self.model.predict(state)  # target_forecast
            target_f = target_f.reshape((9, 9))

            if done:
                target = reward # When the episode is completed (done), the target is set to the reward because there is no subsequent state beyond the achieved one
            if not done:
                # prediction of the Q-values for the next state
                next_q_values = self.model.predict(next_state)
                next_q_values = next_q_values.reshape((9, 9)) # convert to (x,y)
                # Calculation of the target value using the Q-learning formula
                target = reward + self.gamma * np.amax(next_q_values) # Calculates the target based on the Q-learning update rule (only exploration), is the rest of the update rule needed????

            print(f"Vorhergesagter Zielwert: {target_f[action[0], action[1]]}, Berechneter Zielwert: {target}")

            # Updating the Q-value of the executed action
            target_f[action[0], action[1]] = target
            # Training the model with the updated Q-value
            self.model.fit(state, target_f.reshape(1, -1), epochs=1, verbose=0) #fit = trains the model for a fixed number of epochs (here 1)
        # decrease Epsilon
        if self.epsilon > self.exploration_min:
            self.epsilon *= self.exploration_decay # Reduce the epsilon value based on self.epsilon_decay. This decreases over time the rate at which the agent chooses random actions, in favor of exploiting learned knowledge

    def load(self, name):
        """
            Loads the weights of the model from an H5 file.

            Parameters
            ----------
            name:
                path and name of the file from which file it has to be loaded


            Returns
            -------
            None
        """
        self.model.load_weights(name)

    def save_weights(self, name):
        """
            save the weights of the model in a H5 file.

            Parameters
            ----------
            name:
                path and name of the file in which file it should be saved


            Returns
            -------
            None
        """
        self.model.save_weights(name)

    def training(self, env):
        """
            performs the training process.

            Parameters
            ----------
            env:
                Environment to train in

            Returns
            -------
            None
        """
        self.load(path + "2Hole_v2.weights.h5")

        # hyperparameter
        episodes = 1000
        #batch_size = 64

        for episode in range(episodes):
            # initializations for the next episode
            state, _ = env.reset()  # initialize the state of the environment
            total_reward = 0
            done = False
            truncated = False
            batch_size = 0
            self.memory.clear()
            self.memory_important_rewards.clear()

            while not done and not truncated:
                action = self.act(state)  # the agent selects an action based on the state.
                next_state, reward, done, truncated, _ = env.step(action)  # Execution and feedback on the chosen action
                self.remember(state, action, reward, next_state, done)  # Caching the experience
                state = next_state  # update the state for the next iteration
                total_reward += reward  # sum up the reward over the episode
                batch_size += 1
            episode += 1 # increases the episode
            print(f"Episode: {episode}, Reward: {total_reward}")

            """if len(self.memory) >= batch_size:  # Überprüfen, ob genügend Erfahrungen im Speicher sind
                self.train(batch_size)"""
            batch_size = int(batch_size / 5)
            if batch_size > 0 and not truncated: # Only learn if either the goal is reached or the ball is fallen into a hole, meaningful?
                self.train(batch_size) # Trains the neural network based on the stored experiences
            if episode % 10 == 0:  # Every 10 episodes, the agent’s save_weights() method store the model parameters.
                #self.save_weights(output_dir + "episode_" + "{:05d}".format(episode) + ".weights.h5")
                self.save_weights(path + "2Hole_v2" + ".weights.h5")

    def evaluate(self, env):
        """
            performs the trained knowledge.

            Parameters
            ----------
            env:
                Environment to perform the knowledge in

            Returns
            -------
            None
        """
        self.load(path+"2Hole.weights.h5")
        self.epsilon = 0.0 # Pure exploitation of the learned knowledge
        episodes = 10

        for episode in range(episodes):
            state, _ = env.reset()  # initialize the state of the environment
            total_reward = 0
            done = False
            truncated = False

            while not done and not truncated:
                action = self.act(state)  # choose an action based on the current state
                next_state, reward, done, truncated, _ = env.step(action)  # Execute the chosen action
                state = next_state  # updates the state for the next iteration
                total_reward += reward  # sum up the reached reward over an episode
                print(reward)

            episode += 1
            print(f"Episode: {episode}, Reward: {total_reward}")

# -----------------------------------------------------------------------------
# Main (sample)
# -----------------------------------------------------------------------------

if __name__ == '__main__':
    # Init environment and agent
    env = LabyrinthEnvironment(layout='2 holes', render_mode='3D') # evaluate
    #env = LabyrinthEnvironment(layout='2 holes', render_mode=None) # training
    agent = LabyrinthAgentDQN(env.observation_space, env.action_space)
    agent.training(env)
    #agent.evaluate(env)

