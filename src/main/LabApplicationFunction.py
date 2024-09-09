"""
Functions for train and evaluate the agent

@authors: Sandra Lassahn, Marc Hensel
@contact: http://www.haw-hamburg.de/marc-hensel
@copyright: 2024
@version: 2024.05.24
@license: CC BY-NC-SA 4.0, see https://creativecommons.org/licenses/by-nc-sa/4.0/deed.en
"""

#Path to access Environments and Agent
import os
import sys
project_dir = os.path.dirname(os.path.abspath(__file__))
gym_agent_dir = os.path.join(project_dir, '../agents')
sys.path.append(gym_agent_dir)
gym_dir = os.path.join(project_dir, '../environments/virtual')
sys.path.append(gym_dir)
prototype_dir = os.path.join(project_dir, '../environments/physical/Python')
sys.path.append(prototype_dir)

import numpy as np
import torch
import random
import matplotlib.pyplot as plt
from AgentDQN import AgentDQN
from LabyrinthEnv import LabyrinthEnv
from LabyrinthMachine import LabyrinthMachine
import os
import sys

project_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(project_dir, '../models/')
sys.path.append(model_path)
#model_path = '../models/' # Path to load and store weight data


project_dir = os.path.dirname(os.path.abspath(__file__))
prototype_dir = os.path.join(project_dir, '../environments/virtual')
sys.path.append(prototype_dir)
from LabLayouts import Layout

layout_mapping = {
    "HOLES_0_VIRTUAL": Layout.HOLES_0_VIRTUAL,
    "HOLES_2_VIRTUAL": Layout.HOLES_2_VIRTUAL,
    "HOLES_0": Layout.HOLES_0,
    "HOLES_2": Layout.HOLES_2,
    "HOLES_8": Layout.HOLES_8,
    "HOLES_21": Layout.HOLES_21
}

class AppFunction:

    # =========== Constructor =================================================

    def __init__(self, parameter):
        """
            Constructor

            Parameters
            ----------
            parameter: Dict

            Returns
            -------
            None.

        """
        self.__layout = parameter.get("layout")
        self.__layout = layout_mapping.get(self.__layout, self.__layout)
        self.__environment = parameter.get("environment")
        self.__continue_training = parameter.get("continue_training")
        self.__episodes = parameter.get("episodes")
        self.__seed = parameter.get("seed")

        self.__agentsparameter = {
                    "epsilon": parameter.get("epsilon"),
                    "epsilon_decay_rate": parameter.get("epsilon_decay_rate"),
                    "epsilon_min": parameter.get("epsilon_min"),
                    "batch_size": parameter.get("batch_size"),
                    "replay_buffer_size": parameter.get("replay_buffer_size"),
                    "gamma": parameter.get("gamma"),
                    "learning_rate": parameter.get("learning_rate"),
                    "learn_period": parameter.get("learn_period"),
                    "fc1": parameter.get("neurons_layer1"),
                    "fc2": parameter.get("neurons_layer2"),
                    "fc3": parameter.get("neurons_layer3")}

    # -----------------------------------------------------------------------------
    # Evaluation
    # -----------------------------------------------------------------------------

    def __evaluate_init(self):
        """
            initialize the evaluation including the environment and agent

            Parameters
            ----------
            None

            Returns
            -------
            env: virtual or prototype
                virtual: LabyrinthEnv
                prototype: LabyrinthMachine
            agent: AgentDQN
            agent1: (optional)
            agent2: (optional)
        """
        if self.__environment == "Virtual":
            env = LabyrinthEnv(layout=self.__layout, render_mode='3D')  # evaluate simulation
        else:
            env = LabyrinthMachine(layout=self.__layout, cameraID=0)  # evaluate prototype
        agent = AgentDQN(state_size=6, action_size=env.num_actions_per_component * 2, parameter=self.__agentsparameter)
        agent1 = None
        agent2 = None

        if self.__layout == Layout.HOLES_8:
            save_path = model_path + 'HOLES_8_Part1.pth'
            agent.load(save_path)
            agent1 = AgentDQN(state_size=6, action_size=env.num_actions_per_component * 2,
                                       parameter=self.__agentsparameter)
            save_path1 = model_path + 'HOLES_8_Part2.pth'
            agent1.load(save_path1)

        elif self.__layout == Layout.HOLES_21:
            save_path = model_path + 'HOLES_21_Part1.pth'
            agent.load(save_path)
            agent1 = AgentDQN(state_size=6, action_size=env.num_actions_per_component * 2,
                                       parameter=self.__agentsparameter)
            save_path1 = model_path + 'HOLES_21_Part2.pth'
            agent1.load(save_path1)
            agent2 = AgentDQN(state_size=6, action_size=env.num_actions_per_component * 2,
                                       parameter=self.__agentsparameter)
            save_path2 = model_path + 'HOLES_21_Part3.pth'
            agent2.load(save_path2)
        else:
            load_path = model_path + f'{self.__layout.name}.pth'
            agent.load(load_path)

        return env, agent, agent1, agent2


    # =========== Evaluation loop ================================================

    def __evaluate(self, env, agent, agent1, agent2):
        """
            evaluates the agent based on the trained data

            Parameters
            ----------
            env: virtual or prototype
                virtual: LabyrinthEnv
                prototype: LabyrinthMachine
            agent: AgentDQN
            agent1: (optional)
            agent2: (optional)

            Returns
            -------
            None.

        """

        episodes = 10

        scores = []

        for e in range(1, episodes + 1):

            state, _ = env.reset()
            score = 0
            if self.__layout == Layout.HOLES_8 or self.__layout == Layout.HOLES_21:
                progress = 100
                secondpart = False #For undefined tiles in the second part, so that the correct network is still used there and not the first one, because otherwise progress would be a high number.
                thirdpart = False
            while True:
                action = agent.select_action(state, mode='evaluate')  # greedy policy
                if self.__layout == Layout.HOLES_8:
                    if progress < 24 or secondpart == True: #Switching threshold of the trained networks
                        action = agent1.select_action(state, mode='evaluate')
                        secondpart = True
                if self.__layout == Layout.HOLES_21:
                    if progress < 21 or secondpart == True:
                        action = agent1.select_action(state, mode='evaluate')
                        secondpart = True
                    if progress < 10 or thirdpart == True:
                        action = agent2.select_action(state, mode='evaluate')
                        thirdpart = True
                next_state, reward, done, truncated , progress, _ = env.step(action)
                state = next_state
                score += reward
                print(reward)
                if done or score > 4000 or truncated:
                    break

            scores.append(score)

        print(f'Episode {e} Average Score: {np.mean(scores)}')

    # =========== Evaluate_main ==================================================

    def evaluate_main(self):
        """
            main function for evaluation

            Parameters
            ----------
            None.

            Returns
            -------
            None.
        """
        env, agent, agent1, agent2 = self.__evaluate_init()
        self.__evaluate(env, agent, agent1, agent2)

    # -----------------------------------------------------------------------------
    # Training
    # -----------------------------------------------------------------------------

    def __train_init(self):
        """
            initialize the training including the environment and agent

            Parameters
            ----------
            None

            Returns
            -------
            env:
                virtual or prototypical
            agent: AgentDQN
        """
        # Sets the sequence of random numbers to a defined seed.
        seed = self.__seed
        random.seed(seed)
        np.random.seed(seed)  # Seed for NumPy
        torch.manual_seed(seed)  # Seed for PyTorch (CPU)

        # Create environment and agent
        if self.__environment == "Virtual":
            env = LabyrinthEnv(layout=self.__layout, render_mode= None)  # evaluate simulation
        else:
            env = LabyrinthMachine(layout=self.__layout, cameraID=0)  # evaluate prototype

        agent = AgentDQN(state_size = 6, action_size = env.num_actions_per_component * 2, parameter = self.__agentsparameter)
        # loads agents weights if selected
        if self.__continue_training == "Yes" and self.__layout != Layout.HOLES_8 and self.__layout != Layout.HOLES_21:
            save_path = model_path + f'{self.__layout.name}.pth'
            agent.load(save_path)

        return env, agent

    # =========== Training loop ================================================

    def __train(self, env, agent):
        """
            trains the agent

            Parameters
            ----------
            env:
                prototyp: LabyrinthMachine
                virtual: LabyrinthEnv
            agent: AgentDQN

            Returns
            -------
            score: list

        """

        scores = []

        # Train agent
        for e in range(1, self.__episodes + 1):
            state, _ = env.reset()
            score = 0
            agent.decay_epsilon()

            while True:
                action = agent.select_action(state)
                next_state, reward, done, truncated, _, _ = env.step(action)
                agent.step(state, action, reward, next_state, done)
                state = next_state
                score += reward
                if (self.__layout == Layout.HOLES_0_VIRTUAL or self.__layout == Layout.HOLES_0) and (done or truncated or score > 2000):
                    break
                elif done or truncated:
                    break

            print(f'Episode {e} Score: {score}')
            scores.append(score)  # save most recent score
            if e % 10 == 0:
                print(f'Episode {e} Average Score: {np.mean(scores[-100:])}')
            if e % 25 == 0:  # Saves the weights to a different file every 25 episodes.
                save_path = model_path + str(e) + "_" + f'{self.__layout.name}.pth'
                agent.save(save_path)

        return scores

    # =========== Plot results ================================================

    def __plot_scores(self, scores):
        """
            plot the training results

            Parameters
            ----------
            scores: list

            Returns
            -------
            None.

        """
        # Training results scores
        plt.plot(np.arange(len(scores)), scores)
        plt.ylabel('Score')
        plt.xlabel('Episode')
        plt.show()

    # =========== Train_main ==================================================

    def train_main(self):
        """
            main function for training

            Parameters
            ----------
            None.

            Returns
            -------
            None.
        """
        env, agent = self.__train_init()
        scores = self.__train(env=env, agent=agent)
        self.__plot_scores(scores)