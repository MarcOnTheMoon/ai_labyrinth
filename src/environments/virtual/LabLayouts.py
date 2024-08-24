"""
Labyrinth playing field layouts for OpenAI gym environment.

@authors: Sandra Lassahn, Marc Hensel
@contact: http://www.haw-hamburg.de/marc-hensel
@copyright: 2024
@version: 2024.08.25
@license: CC BY-NC-SA 4.0, see https://creativecommons.org/licenses/by-nc-sa/4.0/deed.en
"""
from enum import Enum

class Layout(Enum):
    # Enumeration constants
    HOLES_0      = (0)
    HOLES_0_REAL = (0)
    HOLES_2      = (2)
    HOLES_2_REAL = (2)
    HOLES_8      = (8)
    HOLES_21     = (21)
    
    # ========== Constructor ==================================================
    
    def __init__(self, number_holes):
        """
        Constructor initializing attributes.

        Parameters
        ----------
        number_holes : int
            Number of holes in the labyrinth playing board

        Returns
        -------
        None.

        """
        self.number_holes = number_holes
    

        
