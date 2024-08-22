"""
Main application to train and solve virtual and physical labyrinths.

@authors: Sandra Lassahn, Marc Hensel
@contact: http://www.haw-hamburg.de/marc-hensel
@copyright: 2024
@version: 2024.08.22
@license: CC BY-NC-SA 4.0, see https://creativecommons.org/licenses/by-nc-sa/4.0/deed.en
"""
import os
import sys
project_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(project_dir + '/../agents')
sys.path.append(project_dir + '/../environments/virtual')
sys.path.append(project_dir + '/../environments/physical/Python')

import random
import numpy as np
import matplotlib.pyplot as plt
from LabyrinthEnvironment import LabyrinthEnvironment
from AgentDQN import AgentDQN


# =========== Settings ========================================================

#episodes = 5500
episodes = 2
models_path = '../models/'



class App():
    
    env_types =['virtual', 'physical']

    # =========== Constructor =================================================
    
    def __init__(self, load_model=None, random_seed=None):
        if random_seed != None:
            random.seed(random_seed)
            np.random.seed(random_seed)  # Seed for NumPy
#            torch.manual_seed(random_seed)  # Seed for PyTorch (CPU)

        self.__scores = []

    # =========== Training ====================================================
            
    def train_virtual(self, layout, episodes):
        # Create environment and agent
        env = LabyrinthEnvironment(layout=layout, render_mode=None)
        agent = AgentDQN(state_size = 6, action_size = env.num_actions_per_component * 2)
        
        # Train model
        self.__scores = self.__train(env=env, agent=agent)

    # -------------------------------------------------------------------------
        
    def __train(self, env, agent):
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
#            if e % 25 == 0: #Saves the weights to a different file every 25 episodes.
            if e % 2 == 0: #Saves the weights to a different file every 25 episodes.
                agent.save(models_path + 'model.pth')
        
        return scores


    # =========== Plot results ================================================
    
    def plot_scores(self):
        # Training results scores
        plt.plot(np.arange(len(self.__scores)), self.__scores)
        plt.ylabel('Score')
        plt.xlabel('Episode')
        plt.show()
        

# -----------------------------------------------------------------------------
# Main (sample)
# -----------------------------------------------------------------------------

if __name__ == '__main__':
    app = App()
    app.train_virtual(layout='8 holes', episodes=episodes)
    app.plot_scores()
    
    # Init environment and agent
    #env = LabyrinthEnvironment(layout='0 holes real', render_mode='3D') #train with rendered simulation
    #env = LabyrinthEnvironment(layout='8 holes', render_mode=None) #training simulation
    #env = LabyrinthMachine(layout='0 holes real', cameraID=0) # training device
    #save_path = path + '0holesreal.pth' #Uncomment for further training
    #agent.load(save_path) #Uncomment for further training

