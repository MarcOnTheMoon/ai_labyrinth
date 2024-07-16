"""
Deep Q-Learning (DQN) agent for labyrinth OpenAI gym environment.

@authors: Sandra Lassahn, Marc Hensel
@contact: http://www.haw-hamburg.de/marc-hensel
@copyright: 2024
@version: 2024.05.15
@license: CC BY-NC-SA 4.0, see https://creativecommons.org/licenses/by-nc-sa/4.0/deed.en
"""
# conda install pytorch::pytorch danach bswp in pycharm pip install torch
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

path = "C:/Users/Sandra/Documents/" #lokal Path to load and store weight data

class DqnAgent:

    def __init__(
            self,
            state_size,
            action_size,
            degp_epsilon = 1,
            degp_decay_rate = .98, #für 0 holes .9, für 2 holes .98 und bisherige 8holes
            degp_min_epsilon = .15, #für 0 holes .1, für 2Holes 0.15
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
            state_size:
                size of environment observation space
            action_size:
                size of action space
            degp_epsilon:
                exploration rate ε - initial Epsilon greedy parameter,  1 = explore 100% of the time at the beginning
            degp_decay_rate:
                Rate that reduce chance of random action, decay rate of epsilon greedy policy
            degp_min_epsilon:
                describes how low the exploration rate ε can drop, minimal epsilon value
            train_batch_size:
                batch-Größe für das Training
            replay_buffer_size:
                Größe des Replay-Buffers
            gamma:
                Discount factor for past rewards. gamma -> 1: long-term rewards should be given more consideration. gamma -> 0: short-term, immediate optimization of the reward.
            learning_rate:
                Learning rate for the optimizer
            learn_period:
                wie oft das Q-Netz trainiert werden soll

            Returns
            -------
            None.

        """
        self.state_size = state_size #setzt die Zustandsgröße
        self.action_size = action_size #setzt die Aktionsgröße

        self.degp_epsilon = self.degp_initial_epsilon = degp_epsilon #Setzt den initialen und aktuellen Epsilon-Wert.
        self.degp_decay_rate = degp_decay_rate # Setzt die Abbaurate für Epsilon.
        self.degp_min_epsilon = degp_min_epsilon #Setzt den minimalen Epsilon-Wert.

        # Q-Network initialized in init_q_net method
        self.learn_period = learn_period #Setzt die Häufigkeit des Trainings
        self.learning_rate = learning_rate
        self.init_q_net() # ruft init_q_net auf, um das Q-Netzwerk zu initialisieren.

        # Replay memory
        self.memory = ReplayBuffer(replay_buffer_size, train_batch_size) #Initialisiert den Replay-Buffer mit der angegebenen Größe und Batch-Größe, ruft dafür die init von der klasse ReplayBuffer auf.

        self.training_steps_count = 0 # Initialisiert den Zähler für Trainingsschritte.
        self.train_batch_size = train_batch_size # Setzt die Batch-Größe für das Training.
        self.replay_buffer_size = replay_buffer_size #Setzt die Größe des Replay-Buffers.
        self.gamma = gamma # setzt den Diskonierungsfaktor

        self.error_print_iteration = 0 #zähler fürs ausprinten des lernfehlers

    def init_q_net(self):
        """
            Inizialisierung des Q-Netzes, Optimierers und der Verlustfunktion

            Parameters
            ----------
            None.

            Returns
            -------
            None.

        """
        self.q_net = PtQNet(self.state_size, self.action_size) #initialisiert das Q-Netz, PtQNet ist die Klasse, des neuronalen Netzwerks
        self.optimizer = optim.Adam(self.q_net.parameters(), lr = self.learning_rate) #Optimierer: Adam
        self.loss = torch.nn.MSELoss() # Verlustfunktion: Mean Squared Error (MSE)

    def before_episode(self):
        """
            Anpassung von Epsilon vor einer neuen Episode.

            Parameters
            ----------
            None


            Returns
            -------
            None.

        """
        self.degp_epsilon *= self.degp_decay_rate #Verringert den Epsilon-Wert gemäß der Abbaurate.
        self.degp_epsilon = max(self.degp_epsilon, self.degp_min_epsilon) #Stellt sicher, dass Epsilon nicht unter den minimalen Wert fällt.

    def step(self, state, action, reward, next_state, done):
        """
            speichert Erfahrunngen und löst das Training/Lernen aus

            Parameters
            ----------
            state:
            action:
            reward:
            next_state:
            done:


            Returns
            -------
            None.

        """
        self.memory.add(state, action, reward, next_state, done) # Save experience in replay memory

        self.training_steps_count += 1 # Erhöht den Zähler für Trainingsschritte

        if self.training_steps_count % self.learn_period == 0: # Überprüft, ob es Zeit zum Lernen ist.
            # If enough samples are available in memory, get random subset and learn
            if len(self.memory) > 200: #Überprüft, ob genug Proben im Replay-Buffer sind.
                self.learn() #Ruft die Lernmethode auf.

    def act(self, state, mode = 'train'):
        """
            Auswahl einer Aktion, nach der Epsilon Greedy Policy

            Parameters
            ----------
            state:
            mode:
                mögliche modi: train oder test


            Returns
            -------
            action:
                chose action to rotate the field.

        """
        r = random.random() #Erzeugt eine zufällige Zahl zwischen 0 und 1.
        random_action = mode == 'train' and r < self.degp_epsilon #Überprüft, ob eine zufällige Aktion ausgewählt werden soll (im Trainingsmodus und wenn r kleiner als Epsilon ist).
        if mode == 'test' and r < 0.00: #Zufälligkeit bei der evaluation -> besseren Ergebnissen
            random_action = True
        if random_action: #Wenn eine zufällige Aktion ausgewählt werden soll:
            # Random Policy
            action = random.choice(np.arange(self.action_size)) #Wählt eine zufällige Aktion.
        else: #ansonsten
            # Greedy Policy
            state = torch.from_numpy(state).float().unsqueeze(0)  # Konvertiert den Zustand state von einem NumPy-Array in einen PyTorch-Tensor und fügt eine zusätzliche Dimension hinzu. um aus dem 1D-Zustandsvektor einen 2D-Batch mit einem einzigen Element zu machen. Dies ist wichtig, da neuronale Netze typischerweise Batch-Verarbeitung erwarten.

            self.q_net.eval()  # Schaltet das Netzwerk in den Evaluierungsmodus (deaktiviert Dropout und Batch Normalization).

            with torch.no_grad():  # disabling gradient computation #Deaktiviert die Berechnung der Gradienten, um Speicher zu sparen und die Ausführung zu beschleunigen. Während der Aktionsauswahl benötigen wir keine Gradienten, da wir das Modell nur verwenden, um Vorhersagen zu treffen.
                action_values = self.q_net(state)  # Berechnet die Q-Werte für alle möglichen Aktionen basierend auf dem gegebenen Zustand
            self.q_net.train()  # Schaltet das Netzwerk zurück in den Trainingsmodus, um sicherzustellen, dass es für zukünftige Trainingsschritte bereit ist.
            action = np.argmax(
                action_values.data.numpy())  # Wählt die Aktion mit dem höchsten Q-Wert aus. Dies implementiert eine greedy policy, bei der stets die best bekannte Aktion gewählt wird.

        return action #Gibt die ausgewählte Aktion zurück.

    def learn(self):
        """
            trainieren des Agenten

            Parameters
            ----------
            None

            Returns
            -------
            None

        """
        samples = self.memory.batch() # Entnimmt eine Stichprobe von Erfahrungen aus dem Replay-Speicher.
        s, a, r, s_next, dones = samples # Entpackt die Stichprobe in Zustände s, Aktionen a, Belohnungen r, nächste Zustände s_next und End-Zustände dones.

        # folgende Zeilen konvertieren die NumPy-Arrays in PyTorch-Tensoren:
        s = torch.from_numpy(s).float()
        a = torch.from_numpy(a).long()
        r = torch.from_numpy(r).float()
        s_next = torch.from_numpy(s_next).float()
        dones = torch.from_numpy(dones).float()

        # V(s') = max(Q(s',a))
        v_s_next = self.q_net(s_next).detach().max(1)[0].unsqueeze(1) # Berechnet den maximalen Q-Wert für die nächsten Zustände s_next ohne Gradientenberechnung (mit .detach()) und fügt eine Dimension hinzu.

        # Q(s,a)
        q_sa_pure = self.q_net(s) #Berechnet die Q-Werte für den aktuellen Zustand s.
        q_sa = q_sa_pure.gather(dim = 1, index = a) # Extrahiert die Q-Werte der gewählten Aktionen a aus den berechneten Q-Werten.

        # TD = r + g * V(s') - Q(s,a)
        td = r + (self.gamma * v_s_next * (1 - dones)) - q_sa # Berechnet den Temporal Difference (TD) Fehler gemäß der Aktualisierungsregel des Q-Learnings. self.gamma ist der Diskontierungsfaktor. 1-done wird verwendet, da es nach dem Endzustand keinen nachfolgenden Zustandgibt -> V(s') ist dann 0, sonst V(s')= 1

        # Compute loss: TD -> 0
        error = self.loss(td, torch.zeros(td.shape)) # Berechnet den Verlust zwischen dem TD-Fehler und Null.
        self.error_print_iteration = (self.error_print_iteration + 1) % 100 #alle 100 steps den Fehler ausprinten
        if self.error_print_iteration == 0:
            print(f'error: {error}')
        self.optimizer.zero_grad() # Setzt die Gradienten der Optimierer auf Null zurück. Gradienten werden standardmäßig summiert bei jedem Aufruf von backward(), Wenn die Gradienten nicht auf Null zurückgesetzt werden würden, würden sie sich bei jedem neuen Backpropagation-Schritt zu den vorherigen Gradienten addieren -> falsche Updates.
        error.backward() # Backward= backpropagation: Berechnet die Gradienten des Fehlers bezüglich der Modellparameter.
        self.optimizer.step() # aktualisiert die Netzwerkparameter basierend auf den berechneten Gradienten.


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
    # Zufallsfolge auf definierten Anfang setzen
    seed = 1
    random.seed(seed)
    np.random.seed(seed)  # Seed für NumPy setzen
    torch.manual_seed(seed)  # Seed für PyTorch setzen (CPU)

    # Init environment and agent
    #env = LabyrinthEnvironment(layout='0 holes', render_mode='3D') #evaluate
    env = LabyrinthEnvironment(layout='8 holes', render_mode=None) #training
    agent = DqnAgent(state_size = 6, action_size = env.num_actions_per_component * 2)
    #save_path = path + '2holesreal_dqnagent.pth'
    #agent.load(save_path)
    episodes = 1000
    scores = []
    for e in range(1, episodes + 1):
        state, _ = env.reset()
        score = 0
        agent.before_episode()

        while True:
            action = agent.act(state)

            # next_state, reward, done ,_ = env.step(action) # original
            next_state, reward, done, truncated, _ = env.step(action)
            agent.step(state, action, reward, next_state, done)
            state = next_state
            score += reward
            if done or truncated: # or score > 2000 bei 0 Holes
                break

        print(f'Episode {e} Score: {score}')
        scores.append(score)  # save most recent score
        if e % 10 == 0:
            print(f'Episode {e} Average Score: {np.mean(scores[-100:])}')
        if e % 25 == 0: #alle 200 episoden die Gewichte in einer anderen Datei speichern
            save_path_100 = path + str(e) + '2holesreal_dqnagent.pth'
            agent.save(save_path_100)

    # Training Results scores
    plt.plot(np.arange(len(scores)), scores)
    plt.ylabel('Score')
    plt.xlabel('Episode')
    plt.show()
