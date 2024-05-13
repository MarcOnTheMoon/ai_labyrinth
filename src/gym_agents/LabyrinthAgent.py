#conda install -c conda-forge tensorflow
#conda install pip
#afterwards in anaconda: pip install keras-rl2

# https://keras.io/examples/rl/deep_q_network_breakout/
# https://ebookcentral.proquest.com/lib/hawhamburg-ebooks/reader.action?docID=30168989
# gut    https://domino.ai/blog/deep-reinforcement-learning   Zusammenfassung aus dem Buch: book, Deep Learning Illustrated: A Visual, Interactive Guide to Artificial Intelligence by Krohn, Beyleveld, and Bassens.
# gut    https://medium.com/@gtnjuvin/my-journey-into-deep-q-learning-with-keras-and-gym-3e779cc12762
# !!!    https://www.tensorflow.org/agents/tutorials/1_dqn_tutorial
# https://stable-baselines3.readthedocs.io/en/master/modules/dqn.html

import random
import gym
import numpy as np
from collections import deque
from tensorflow.keras import Sequential, Input
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers import Adam
#from rl.policy import EpsGreedyQPolicy
#from rl.memory import SequentialMemory
#from rl.agents.dqn import DQNAgent

from LabyrinthEnvironment import LabyrinthEnvironment

class DQNAgent:
    def __init__(self, state_space, action_space):

        self.state_space = state_space
        self.action_space = action_space
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95  # Discount factor for past rewards
        self.epsilon = 1.0  # exploration rate - Epsilon greedy parameter
        self.exploration_min = 0.01
        self.exploration_decay = 0.99  # Rate at which to reduce chance of random action being taken
        self.learning_rate = 0.001

        self.model = self._build_model()

    def _build_model(self):
        model = Sequential() # Sequential a simple stack of layers (one input, one output), alterntive Functional Input???
        #input_shape = self.state_space['ball_position'].shape[0] + self.state_space['ball_velocity'].shape[0] + self.state_space['field_rotation'].shape[0]

        #model.add(Flatten(input_shape=(1, input_shape)))
        model.add(Input(shape=(6,), dtype='float32', name='state')) #jetzt überhaupt sequenzial benötigt?
        model.add(Dense(64, activation='relu')) # Dense is the basic form of a neural network layer
        model.add(Dense(64, activation='relu'))
        model.add(Dense(1, activation='linear', name='action'))
        model.summary()
        model.compile(loss='mse', optimizer=Adam(learning_rate=self.learning_rate))
        return model

    """def _build_agent(self, model, action):
    # werte in model einbauen vor compile?
        policy = EpsGreedyQPolicy
        memory = SequentialMemory(limit=2000, window_length=1)
        agent = DQNAgent(model=model, memory=memory, policy=policy, nb_actions=action, nb_steps_warmup=10, target_model_update=1e-2)
        return agent"""

    def remember(self, state, action, reward, next_state, done): # simply store states, actions and resulting rewards into the memory
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return self.action_space.sample()
            #return random.randrange(self.action_space)
        q_values = self.model.predict(state)
        action = np.argmax(q_values[0])
        return action

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size) #only take a few samples and we will just pick them randomly.
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = reward + self.gamma * np.amax(self.model.predict(next_state)[0]) #model oder agent?
            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0) #model oder agent?
        if self.epsilon > self.exploration_min:
            self.epsilon *= self.exploration_decay

    def load(self, name):
        self.model.load_weights(name)

    def save_weights(self, name):
        self.model.save_weights(name) #save Q Network parameters to a file

if __name__ == '__main__':
    env = LabyrinthEnvironment(render_mode='3D')  # Init gym environment
    agent = DQNAgent(env.observation_space, env.action_space)


    episodes = 500 # Anzahl der Episoden zum Trainieren des Agenten
    batch_size = 32 # Größe der Minibatches für das Training

    #training
    for episode in range(episodes):
        state, _ = env.reset()  # Initialisiert den Zustand der Umgebung
        total_reward = 0
        done = False

        while not done:
            action = agent.act(state)  # Der Agent wählt eine Aktion basierend auf dem Zustand
            env.render(action)  # Die Umgebung führt die ausgewählte Aktion aus
            for i in range(8):
                env.render_ball(action) # Ballbewegung ausführen ein paar mal ausführen Rückmeldung zur Aktion
            next_state, reward, done, _ = env.step(action) # Rückmeldung über die getätigte Aktion
            agent.remember(state, action, reward, next_state, done)  # Der Agent speichert die Erfahrung
            state = next_state  # Aktualisiert den Zustand für die nächste Iteration
            total_reward += reward  # Summiert die Belohnung über die Episode
        episode += 1
        print(f"Episode: {episode}, Reward: {total_reward}")

        if len(agent.memory) >= batch_size:  # Überprüfen, ob genügend Erfahrungen im Speicher sind
            agent.replay(batch_size)  # Der Agent führt das Training basierend auf den gespeicherten Erfahrungen durch

    agent.save_weights('dqn_weights.h5')