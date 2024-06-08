import numpy as np
import random
from collections import namedtuple, deque


class ReplayBuffer:

    def __init__(self, buffer_size, batch_size):
        """
            Constructor.

            Parameters
            ----------
            buffer_size:
                maximale Größe des puffers
            batch_size:
                größe der Trainingsbatches

            Returns
            -------
            None.

        """
        self.memory = deque(maxlen = buffer_size) #Initialisiert den Puffer self.memory mit einer maximalen Länge von buffer_size. deque ermöglicht das effiziente Hinzufügen und Entfernen von Elementen.
        self.batch_size = batch_size # Setzt die Batch-Größe für das Training.
        self.experience = namedtuple("Experience", field_names = ["state", "action", "reward", "next_state", "done"]) #Definiert eine benannte Tupelklasse Experience mit den Feldern state, action, reward, next_state und done. dient zu strukturierung der gespeicherten Erfahrungen

    def add(self, state, action, reward, next_state, done):
        """
            Add a new experience to the memory

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
        e = self.experience(state, action, reward, next_state, done) #Erstellt eine neue Instanz des Experience-Tupels mit den neuen Daten.
        self.memory.append(e) #Fügt die neue Erfahrung e dem Speicher self.memory hinzu. Wenn der Puffer voll ist, wird das älteste Element entfernt.

    def batch(self, batch_size = None):
        """
            Randomly sample a batch of experiences from the memory

            Parameters
            ----------
            batch_size: optional
                Größe der Trainingsbatches

            Returns
            -------
            states: NumPy-Array
                alle states der ausgewälten erfahrungen
            actions: NumPy-Array
                alle actions der ausgewälten erfahrungen
            rewards: NumPy-Array
                alle rewards der ausgewälten erfahrungen
            next_states: NumPy-Array
                alle next states der ausgewälten erfahrungen
            dones: NumPy-Array in uint8
                alle dones der ausgewälten erfahrungen

        """
        if batch_size is None:
            batch_size = self.batch_size #Wenn keine Batch-Größe angegeben wurde, wird die Standard-Batch-Größe verwendet.

        experiences = random.sample(self.memory, k = batch_size) #Wählt zufällig batch_size Erfahrungen aus dem Speicher self.memory

        states = np.vstack([e.state for e in experiences if e is not None]) #Erstellt ein NumPy-Array, das alle state-Werte der ausgewählten Erfahrungen enthält.
        actions = np.vstack([e.action for e in experiences if e is not None])
        rewards = np.vstack([e.reward for e in experiences if e is not None])
        next_states = np.vstack([e.next_state for e in experiences if e is not None])
        dones = np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8) # Erstellt ein NumPy-Array, das alle done-Werte der ausgewählten Erfahrungen enthält und konvertiert diese in uint8 (0 oder 1).

        return states, actions, rewards, next_states, dones #Gibt die Batches von Zuständen, Aktionen, Belohnungen, nächsten Zuständen und done-Flags zurück.

    def __len__(self): #ohne len gehts nicht -> Fehlermeldung!!!
        #Return the current size of internal memory.
        return len(self.memory) #Gibt die Anzahl der Elemente im Speicher self.memory zurück.
