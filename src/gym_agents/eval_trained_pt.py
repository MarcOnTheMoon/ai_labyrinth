import os
import sys
import numpy as np
from LabyrinthAgentDQN import DqnAgent

#Path to access LabyrinthEnvironment
project_dir = os.path.dirname(os.path.abspath(__file__))
gym_dir = os.path.join(project_dir, '../gym')
sys.path.append(gym_dir)
from LabyrinthEnvironment import LabyrinthEnvironment

path = "C:/Users/Sandra/Documents/" #lokal Path to load and store weight data


# PyTorch Implementation
if __name__ == '__main__':
    env = LabyrinthEnvironment(layout='8 holes', render_mode='3D')  # evaluate
    agent = DqnAgent(state_size = 6, action_size = 10)
    save_path = path + '0holes_dqnagent.pth'

    agent.load(save_path)

    episodes = 10

    scores = []

    for e in range(1, episodes + 1):

        state, _ = env.reset()
        #state = state[0] #env.reset liefert einen tupel und kein Array, konvertieren ins richtige format

        score = 0

        while True:
            env.render()
            action = agent.act(state, mode = 'test') #greedy policy
            next_state, reward, done, _ ,_ = env.step(action)
            state = next_state
            score += reward
            if done:
                break

        scores.append(score)

    print(f'Episode {e} Average Score: {np.mean(scores)}')
