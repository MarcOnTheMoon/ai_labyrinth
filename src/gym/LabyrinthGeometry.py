"""
Geometric objects for a labyrinth OpenAI gym environment.

Lengths are stated without unit, but are interpreted as [cm].

@authors: Marc Hensel, Sandra Lassahn
@contact: http://www.haw-hamburg.de/marc-hensel
@copyright: 2024
@version: 2024.03.30
@license: CC BY-NC-SA 4.0, see https://creativecommons.org/licenses/by-nc-sa/4.0/deed.en
"""

# -----------------------------------------------------------------------------
# Class containing individual components of pinball geometry
# -----------------------------------------------------------------------------

class LabyrinthGeometry:
    """
    Representation of complete labyrinth geometry.
    """
    def __init__(self):
        """
        Constructor.

        Returns
        -------
        None.

        """
        self.box = Box()
        self.field = Field()
        self.ball = Ball()
    
# -----------------------------------------------------------------------------
# Playing field
# -----------------------------------------------------------------------------
        
class Field:
    """
    Representation of playing field.
    
    The field is represented by its dimensions and rotations around the x- and
    y-axes.
    """
    def __init__(self, size_x=27.4, size_y=22.8, plate_depth=0.3):
        """
        Constructor.
        
        Rotations are initialized with 0 degree.

        Parameters
        ----------
        size_x : float, optional
            Width of the field in [cm]. The default is 27.4.
        size_y : float, optional
            Height of the field in [cm]. The default is 22.8.
        plate_depth : float, optional
            Thickness of the field's plate in [cm]. The default is 0.3.

        Returns
        -------
        None.

        """
        # Dimension
        self.size_x = size_x
        self.size_y = size_y
        self.plate_depth = plate_depth
        
        # Rotation angles
        self.rotation_x_deg = 0.0
        self.rotation_y_deg = 0.0

# -----------------------------------------------------------------------------
# Box with knobs to tilt field
# -----------------------------------------------------------------------------

class Box:
    """
    Representation of the box housing the tiltable field.
    """
    def __init__(self, height=9.5, boarder=0.8, wheel_radius=1.7, wheel_depth=1.5):
        """
        Constructor.

        Parameters
        ----------
        height : float, optional
            Height in [cm]. The default is 9.5.
        boarder : float, optional
            Thickness of the boxes plates in [cm]. The default is 0.8.
        wheel_radius : float, optional
            Radius of the knobs operating the game in [cm]. The default is 1.7.
        wheel_depth : float, optional
            Depth of the knobs operating the game in [cm]. The default is 1.5.

        Returns
        -------
        None.

        """
        self.height = height
        self.boarder = boarder
        self.wheel_radius = wheel_radius
        self.wheel_depth = wheel_depth

# -----------------------------------------------------------------------------
# Ball
# -----------------------------------------------------------------------------

class Ball:
    """
    Representation of ball.
    
    The default values correspond to a ball from the game series Labyrinth and
    GraviTrax, having a diameter of 12.7 mm and a weight of 9 g.
    """
    def __init__(self, radius=0.635, weightGram=9.0):
        """
        Constructor.

        Parameters
        ----------
        radius : float, optional
            Raidus in [cm]. The default is 0.635.
        weightGram : float, optional
            Weight in [g]. The default is 9.0.

        Returns
        -------
        None.

        """
        self.radius = radius
        self.weightGram = weightGram
