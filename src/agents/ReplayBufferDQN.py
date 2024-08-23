"""
Represents the replay buffer for the DQN agent.

@authors: Sandra Lassahn, Marc Hensel
@contact: http://www.haw-hamburg.de/marc-hensel
@copyright: 2024
@version: 2024.08.23
@license: CC BY-NC-SA 4.0, see https://creativecommons.org/licenses/by-nc-sa/4.0/deed.en
"""
import numpy as np
import random
from collections import namedtuple, deque

class ReplayBuffer:

    # ========== Constructor ==================================================

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
        # TODO Why use deque (add/remove at left and right => FIFO and LIFO) instead of queue (FIFO)? Is queue faster?
        self.__memory = deque(maxlen = buffer_size)     # Deque allows efficient addition and removal of elements
        self.__batch_size = batch_size
        self.__experience = namedtuple('Experience', field_names = ['state', 'action', 'reward', 'next_state', 'done'])

    # ========== Add element (experience) =====================================

    def add(self, state, action, reward, next_state, done):
        """
        Append a new experience to the memory.
        
        Removes the oldest element, if the buffer is full.

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
        # Append element
        element = self.__experience(state, action, reward, next_state, done)
        self.__memory.append(element)

        # Re-save target experiences to increase the likelihood of learning more from them
        # TODO Value 300 arbitrary? Why handled by buffer data type, not by agent?
        if reward > 300:
            self.__memory.append(element)

    # ========== Add element (experience) =====================================

    def get_random_batch(self):
        """
        Randomly sample a batch of experiences from the memory

        Parameters
        ----------
        None

        Returns
        -------
        states: NumPy array
            All states of the selected experiences
        actions: NumPy array
            All actions of the selected experiences
        rewards: NumPy array
            All rewards of the selected experiences
        next_states: NumPy array
            All next_states of the selected experiences
        dones: NumPy array in uint8
            All dones of the selected experiences coded as 0 (False) or 1 (True)

        """
        # Get batch_size random experiences from the buffer
        experiences = random.sample(self.__memory, k = self.__batch_size)

        # Create NumPy arrays of entries within batch of experiences
        # TODO Faster to have 5 queues or deques (i.e., for states, actions, and so on, each)?
        states = np.vstack([e.state for e in experiences if e is not None])
        actions = np.vstack([e.action for e in experiences if e is not None])
        rewards = np.vstack([e.reward for e in experiences if e is not None])
        next_states = np.vstack([e.next_state for e in experiences if e is not None])
        dones = np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)

        return states, actions, rewards, next_states, dones

    # ========== Implement len() method =======================================
    
    def __len__(self):
        """
        Implement the len() method.
        
        Note that the method returns the number of experiences currently
        stored in the buffer, NOT the capacity (i.e., not maxlen of the dequeue).

        Parameters
        ----------
        None

        Returns
        -------
        Number of experiences stored
            
        """
        # TODO Where used? (Original comment was: "Error messages without")
        return len(self.__memory)
