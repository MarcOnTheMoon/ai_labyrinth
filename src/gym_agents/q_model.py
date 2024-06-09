import torch.nn as nn
import torch.nn.functional as F


class PtQNet(nn.Module): # PtQNet, die von nn.Module erbt. nn.Module ist die Basisklasse für alle neuronalen Netzwerke in PyTorch
    """Q-Network"""

    def __init__(self, state_size, action_size, fc1_units = 128, fc2_units = 128, fc3_units = 64):
        """
            Constructor. initialisiert die Netzwerkschichten

            Parameters
            ----------
            state_size:
                Dimension of the environment observation space
            action_size:
                Dimension of action space
            fc1_units:
                First hidden layer size
            fc2_units:
                Second hidden layer size
            fc3_units:
                Third hidden layer size

            Returns
            -------
            None.

        """
        super(PtQNet, self).__init__() #Ruft den Konstruktor der Basisklasse (nn.Module) auf. Dies ist notwendig, um die Vererbung korrekt zu initialisieren.
        self.relu = nn.ReLU() #Initialisiert die ReLU-Aktivierungsfunktion (Rectified Linear Unit)
        self.fc1 = nn.Linear(state_size, fc1_units) #Definiert die erste vollverbundene Schicht (Linear Layer) mit state_size Eingabeneuronen und fc1_units Ausgabeneuronen.
        self.bn1 = nn.BatchNorm1d(fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units) # Definiert die zweite vollverbundene Schicht mit fc1_units Eingabeneuronen (von der ersten Schicht) und fc2_units Ausgabeneuronen.
        self.bn2 = nn.BatchNorm1d(fc2_units)
        #self.fc3 = nn.Linear(fc2_units, fc3_units)
        self.fc3 = nn.Linear(fc2_units, action_size) #Definiert die dritte vollverbundene Schicht mit fc2_units Eingabeneuronen und action_size Ausgabeneuronen. Diese Schicht gibt die Q-Werte für jede mögliche Aktion aus.

    def forward(self, state):
        """
            Definiert den Vorwärtsdurchlauf durch das Netzwerk. Es findet die Berechnung der Ausgabe des Netzwerks (Q-Werte) basierend eine gegebene Eingabe (state) statt.

            Parameters
            ----------
            state:
                environment observation space


            Returns
            -------
                self.fc3(x):
                    berechnete Q-Werte für den Eingabezustand

        """
        x = F.leaky_relu(self.bn1(self.fc1(state))) #Der Eingabezustand (state) wird durch die erste vollverbundene Schicht fc1 geleitet und dann durch die ReLU-Aktivierungsfunktion. Das Ergebnis wird in x gespeichert.
        x = F.leaky_relu(self.bn2(self.fc2(x))) # Die Ausgabe der ersten Schicht wird durch die zweite vollverbundene Schicht fc2 geleitet und dann erneut durch die ReLU-Aktivierungsfunktion. Das Ergebnis wird wieder in x gespeichert.
        #x = self.relu(self.fc3(x))
        return self.fc3(x) #Die Ausgabe der zweiten Schicht wird durch die dritte vollverbundene Schicht fc3 geleitet. Diese Schicht hat keine Aktivierungsfunktion, da sie die endgültigen Q-Werte für jede Aktion ausgibt.
