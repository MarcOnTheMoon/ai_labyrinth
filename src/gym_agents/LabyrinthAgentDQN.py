"""
Deep Q-Learning (DQN) agent for labyrinth OpenAI gym environment.

@authors: Sandra Lassahn, Marc Hensel
@contact: http://www.haw-hamburg.de/marc-hensel
@copyright: 2024
@version: 2024.05.15
@license: CC BY-NC-SA 4.0, see https://creativecommons.org/licenses/by-nc-sa/4.0/deed.en
"""
#conda install -c conda-forge tensorflow
#conda install pip
#afterwards in anaconda: pip install keras-rl2


import random
import numpy as np
from collections import deque
from tensorflow.keras import Sequential, Input
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers import Adam

#Pfad damit auf LabyrinthEnvironment zugegriffen werden kann
import sys
import os
project_dir = os.path.dirname(os.path.abspath(__file__))
gym_dir = os.path.join(project_dir, '../gym')
sys.path.append(gym_dir)
from LabyrinthEnvironment import LabyrinthEnvironment

path = "C:/Users/Sandra/Documents/" #Pfad zum speichern und laden der h5 datein in der die gewichte gespeichter werden

class LabyrinthAgentDQN:
    def __init__(self, state_space, action_space):

        self.state_space = state_space
        self.action_space = action_space
        self.memory = deque(maxlen=2000) #Die Erinnerungen werden als Elemente einer Datenstruktur namens "deque" („deck“) gespeichert, die ähnlich wie eine Liste funktioniert. Behält nur die neusten maxlength Elemente
        #self.memory_important_rewards = deque(maxlen=20) #positive und hohe negative belohnungen speichern
        self.gamma = 0.95  # Discount factor for past rewards
        self.epsilon = 1.0  # exploration rate - Epsilon greedy parameter,  1.0 = zu Beginn 100% der Zeit erkunden zu lassen
        self.exploration_min = 0.01 # beschreibt wie niedrig die Explorationsrate ε abfallen kann
        self.exploration_decay = 0.99  # Rate at which to reduce chance of random action being taken as the agent gets better and better in playing
        self.learning_rate = 0.001 #stochastische Gradientenabstiegs-Hyperparameter

        self.model = self._build_model()

    def _build_model(self):
        #Neuronales Netz aufbauen
        model = Sequential()
        model.add(Input(shape=(6,), dtype='float32', name='state')) #Eingabeschicht
        #model.add(Dense(32, input_dim=6, activation='relu')) # Eingabeschicht & verdeckte schicht 32
        model.add(Dense(64, activation='sigmoid')) # Dense is the basic form of a neural network layer
        model.add(Dense(64, activation='sigmoid'))
        model.add(Dense(9*9, activation='softmax', name='action')) #Ausgabeschicht
        model.summary()  # Zeigt eine Zusammenfassung des Modells an, einschließlich der Anzahl der Parameter pro Schicht
        model.compile(loss='mse', optimizer=Adam(learning_rate=self.learning_rate))  #mse = mean_squared_error, auch mae = mean_absolut_error oder mean_q = durchschnittlicher Q-Wert
        return model

    def remember(self, state, action, reward, next_state, done): # simply store states, actions and resulting rewards into the memory
        #Replay Memory wird genutzt, um Erfahrungen zu speichern und später zum Training zu verwenden.
        self.memory.append((state, action, reward, next_state, done)) # memory dient zum Speichern von Erinnerungen, die anschließend abgespielt werden können, um das neuronale Netz das DQN zu trainieren
        #if reward > 0 or reward < -100:
            #self.memory_important_rewards.append((state, action, reward, next_state, done)) #merkt sich die wichtigen ereignisse (positive belohnungen und löcher)

    def act(self, state):
        #Epsilon-Greedy-Policy für die Aktionsauswahl
        if np.random.rand() <= self.epsilon:
            # Take random action
            x_action = np.random.choice([0,1,2,3,4,5,6,7,8])
            y_action = np.random.choice([0,1,2,3,4,5,6,7,8])
            return [x_action, y_action]
        else:
            state = np.array(state).reshape(1, -1)
            q_values = self.model.predict(state) # predict(): Modell sagt die Belohnung des aktuellen Zustands basierend auf den bis dato trainierten Daten vorher
            act_values = q_values.reshape((9, 9)) # convert to a 2D array representing the (x, y) action grid
            x_action, y_action = np.unravel_index(np.argmax(act_values), act_values.shape) # Take best action mit konvertierung in die diemnsion der (x, y) Koordinaten
            return [x_action, y_action]

    def train(self, batch_size):
        # Diese Funktion trainiert das neuronale Netz mit zufälligen Stichproben aus dem Speicher.
        minibatch = random.sample(self.memory, batch_size) #only take a few samples (batch_size) out of self.memory, pick them randomly.
        #if len(self.memory_important_rewards) > 0:
         #   for state, action, reward, next_state, done in self.memory_important_rewards:
          #      minibatch.append((state, action, reward, next_state, done)) #wichtige ereignisse immer lernen/speichern
        for state, action, reward, next_state, done in minibatch:
            target = reward # if done Wenn die Episode abgeschlossen ist (done), wird das Ziel (target) auf die Belohnung (reward) gesetzt.
            if not done:
                next_state = np.array(next_state).reshape(1, -1)
                next_state = self.model.predict(next_state)
                next_state = next_state.reshape((9, 9)) # in (x,y) konvertieren
                target = reward + self.gamma * np.amax(next_state) #berechnet das Ziel (target) basierend auf dem Q-Learning-Update-Regel (Erkundung)

            state = np.array(state).reshape(1, -1)
            target_f = self.model.predict(state) # Sagt die Q-Werte für den aktuellen Zustand (state) voraus.
            target_f = target_f.reshape((9, 9))
            target_f[action[0], action[1]] = target #Aktualisiert den Q-Wert der ausgeführten Aktion (action) mit dem berechneten Zielwert (target)
            target_f = target_f.flatten()
            self.model.fit(state, target_f.reshape(1, -1), epochs=1, verbose=0) #fit = trains the model for a fixed number of epochs (hier 1)
        if self.epsilon > self.exploration_min:
            self.epsilon *= self.exploration_decay # Reduziert den epsilon-Wert basierend auf self.epsilon_decay. Dies reduziert im Laufe der Zeit die Rate, mit der der Agent zufällige Aktionen wählt, zugunsten der Nutzung des erlernten Modells.

    def load(self, name):
        self.model.load_weights(name) #Loads the weights of the DQN agent from an H5 file.

    def save_weights(self, name):
        self.model.save_weights(name) #Saves the weights of the DQN agent in an H5 file.

    def training(self, env):
        #self.load(path + "8Hole_v3.weights.h5")

        # Hyperparameters
        episodes = 1000
        batch_size = 64

        for episode in range(episodes):
            state, _ = env.reset()  # Initialisiert den Zustand der Umgebung
            total_reward = 0
            done = False
            truncated = False

            while not done and not truncated:
                action = self.act(state)  # Der Agent wählt eine Aktion basierend auf dem Zustand
                next_state, reward, done, truncated, _ = env.step(action)  # Rückmeldung über die getätigte Aktion
                self.remember(state, action, reward, next_state, done)  # Der Agent speichert die Erfahrung
                state = next_state  # Aktualisiert den Zustand für die nächste Iteration
                total_reward += reward  # Summiert die Belohnung über die Episode
            episode += 1
            print(f"Episode: {episode}, Reward: {total_reward}")

            if len(self.memory) >= batch_size:  # Überprüfen, ob genügend Erfahrungen im Speicher sind
                self.train(
                    batch_size)  # Der Agent führt das Training basierend auf den gespeicherten Erfahrungen durch
            if episode % 20 == 0:  # Every 50 episodes, the agent’s save_weights() method store the neural net model’s parameters.
                #self.save_weights(output_dir + "episode_" + "{:05d}".format(episode) + ".weights.h5")
                self.save_weights(path + "8Hole_v3" + ".weights.h5")

    def evaluate(self, env):
        self.load(path+"8Hole_v3.weights.h5")
        self.epsilon = 0.0 #reine ausbeutung des erlernten
        episodes = 10

        for episode in range(episodes):
            state, _ = env.reset()  # Initialisiert den Zustand der Umgebung
            total_reward = 0
            done = False

            while not done:
                action = self.act(state)  # Der Agent wählt eine Aktion basierend auf dem Zustand
                next_state, reward, done, truncated, _ = env.step(action)  # Rückmeldung über die getätigte Aktion
                state = next_state  # Aktualisiert den Zustand für die nächste Iteration
                total_reward += reward  # Summiert die Belohnung über die Episode
            episode += 1
            print(f"Episode: {episode}, Reward: {total_reward}")

# -----------------------------------------------------------------------------
# Main (sample)
# -----------------------------------------------------------------------------

if __name__ == '__main__':
    # Init environment and agent
    #env = LabyrinthEnvironment(layout='8 holes', render_mode='3D') #evaluate
    env = LabyrinthEnvironment(layout='8 holes', render_mode=None) #training
    agent = LabyrinthAgentDQN(env.observation_space, env.action_space)
    agent.training(env)
    #agent.evaluate(env)

