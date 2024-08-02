"""
Represents the Replay-Buffer for the DQN Agent.

@authors: Sandra Lassahn, Marc Hensel
@contact: http://www.haw-hamburg.de/marc-hensel
@copyright: 2024
@version: 2024.08.01
@license: CC BY-NC-SA 4.0, see https://creativecommons.org/licenses/by-nc-sa/4.0/deed.en
"""

import numpy as np
import random
from collections import namedtuple, deque


class LabyrinthReplayBuffer:

    def __init__(self, buffer_size, batch_size):
        """
            Constructor.

            Parameters
            ----------
            buffer_size: int
                Maximum size of the replay buffer
            batch_size: int
                Size of training batches

            Returns
            -------
            None.

        """
        self.__memory = deque(maxlen = buffer_size) #Initializes the buffer self.__memory with a maximum length of buffer_size. deque allows efficient addition and removal of elements.
        self.__batch_size = batch_size # Sets the batch size for training.
        self.__experience = namedtuple("Experience", field_names = ["state", "action", "reward", "next_state", "done"]) #Defines a named tuple class Experience with the fields state, action, reward, next_state, and done. It's used to structure the stored experiences.

    def add(self, state, action, reward, next_state, done):
        """
            Add a new experience to the memory

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
        e = self.__experience(state, action, reward, next_state, done) #Creates a new instance of the Experience tuple with the new data.
        self.__memory.append(e) #Adds the new experience e to the self.__memory buffer. If the buffer is full, the oldest element is removed

        if reward > 300:  # Re-save target experiences to increase the likelihood of learning more from them
            self.__memory.append(e)

    def batch(self):
        """
            Randomly sample a batch of experiences from the memory

            Parameters
            ----------
            None

            Returns
            -------
            states: NumPy-Array
                All states of the selected experiences
            actions: NumPy-Array
                All actions of the selected experiences
            rewards: NumPy-Array
                All rewards of the selected experiences
            next_states: NumPy-Array
                All next_states of the selected experiences
            dones: NumPy-Array in uint8
                All dones of the selected experiences

        """

        experiences = random.sample(self.__memory, k = self.__batch_size) #Randomly selects batch_size experiences from the self.__memory buffer.

        states = np.vstack([e.state for e in experiences if e is not None]) #Creates a NumPy array containing all the state values of the selected experiences.
        actions = np.vstack([e.action for e in experiences if e is not None])
        rewards = np.vstack([e.reward for e in experiences if e is not None])
        next_states = np.vstack([e.next_state for e in experiences if e is not None])
        dones = np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8) #Creates a NumPy array containing all the done values of the selected experiences and converts them to uint8 (0 or 1).

        return states, actions, rewards, next_states, dones #Returns the batch-size experiences of states, actions, rewards, next states, and done flags.

    def __len__(self): #Without len, it doesn't work – error message!
        """
            Return the current size of internal memory, from deque.

            Parameters
            ----------
            None

            Returns
            -------
            int
        """
        return len(self.__memory)
