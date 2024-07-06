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
    agent = DqnAgent(state_size = 6, action_size = env.num_actions_per_component * 2)
    save_path = path + '8holes_dqnagent_part1.pth'
    agent.load(save_path)
    if env.layout == '8 holes':
        agent1 = DqnAgent(state_size=6, action_size=env.num_actions_per_component * 2)
        save_path1 = path + '8holes_dqnagent_part2.pth'
        agent1.load(save_path1)

    episodes = 10

    scores = []

    for e in range(1, episodes + 1):

        state, _ = env.reset()
        #state = state[0] #env.reset liefert einen tupel und kein Array, konvertieren ins richtige format

        score = 0
        if env.layout == '8 holes':
            progress = 100
            secondpart = False #für nicht definierte Kacheln im zweiten Teil, so dass dort trotzdem das richtige Netzwerk verwent wird und nciht das erste weil progress dann eine hohe zahl wäre
        while True:
            env.render()
            action = agent.act(state, mode='test')  # greedy policy
            if env.layout == '8 holes':
                if progress < 24 or secondpart == True: #umschaltschwelle der trainierten Netze
                    action = agent1.act(state, mode='test')
                    secondpart = True
            next_state, reward, done, _ , progress = env.step(action)
            state = next_state
            score += reward
            print(reward)
            if done or score > 5000:
                break

        scores.append(score)

    print(f'Episode {e} Average Score: {np.mean(scores)}')
