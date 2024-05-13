"""
Geometric objects for a labyrinth OpenAI gym environment.

Lengths are stated without unit, but are interpreted as [cm].

@authors: Marc Hensel, Sandra Lassahn
@contact: http://www.haw-hamburg.de/marc-hensel
@copyright: 2024
@version: 2024.04.15
@license: CC BY-NC-SA 4.0, see https://creativecommons.org/licenses/by-nc-sa/4.0/deed.en
"""
from vpython import vector as vec

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
        self.walls = Walls()
        self.holes = Holes()
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
# fieldwalls
# -----------------------------------------------------------------------------

class Walls:

    def __init__(self, walls_data=None, walls_thickness=0.58, walls_width=0.6):
        z_pos = walls_width / 2 #z-position wall
        if walls_data is None:
            # walls (field with 2 holes)
            """walls_data = [
                {"pos": vec(-3.41, 7.52, z_pos), "size": vec(walls_thickness, 7.59, walls_width)},
                {"pos": vec(-2.37, 4, z_pos), "size": vec(2.64, walls_thickness, walls_width)},
                {"pos": vec(-7.78, -4.54, z_pos), "size": vec(11.83, walls_thickness, walls_width)},
                {"pos": vec(-2.14, -7.82, z_pos), "size": vec(walls_thickness, 7.12, walls_width)},
                {"pos": vec(-0.13, -6.9, z_pos), "size": vec(4.45, walls_thickness, walls_width)},

                {"pos": vec(1.84, -7.82, z_pos), "size": vec(walls_thickness, 7.12, walls_width)},
                {"pos": vec(7.53, -4.54, z_pos), "size": vec(11.95, walls_thickness, walls_width)},
                {"pos": vec(3.61, -1.77, z_pos), "size": vec(walls_thickness, 6.05, walls_width)},
            ]"""

            #walls (field with 8 holes)
            walls_data = [
                {"pos": vec(2.01, 9.09, z_pos), "size": vec(walls_thickness, 4.54, walls_width)},
                {"pos": vec(-4.08, 9.49, z_pos), "size": vec(11.72, walls_thickness, walls_width)},
                {"pos": vec(-9.66, 9.09, z_pos), "size": vec(walls_thickness, 1.42, walls_width)},
                {"pos": vec(2.82, 7.1, z_pos), "size": vec(2.04, walls_thickness, walls_width)},
                {"pos": vec(-11.57, 5.65, z_pos), "size": vec(walls_thickness, 2.36, walls_width)},

                {"pos": vec(-6.92, 5.85, z_pos), "size": vec(9.85, walls_thickness, walls_width)},
                {"pos": vec(-2.28, 0.96, z_pos), "size": vec(walls_thickness, 10.32, walls_width)},
                {"pos": vec(0.44, 4, z_pos), "size": vec(5.92, walls_thickness, walls_width)},
                {"pos": vec(-3.62, -3.96, z_pos), "size": vec(11.17, walls_thickness, walls_width)},
                {"pos": vec(-3.32, -5.18, z_pos), "size": vec(walls_thickness, 3.01, walls_width)},

                {"pos": vec(-8.96, -4.85, z_pos), "size": vec(walls_thickness, 3.92, walls_width)},
                {"pos": vec(-9.82, -6.55, z_pos), "size": vec(2.28, walls_thickness, walls_width)},
                {"pos": vec(-9.19, 0.11, z_pos), "size": vec(8.73, walls_thickness, walls_width)},
                {"pos": vec(-10.11, 0.91, z_pos), "size": vec(walls_thickness, 2.13, walls_width)},
                {"pos": vec(-6.51, 0.91, z_pos), "size": vec(walls_thickness, 2.13, walls_width)},

                {"pos": vec(-12.83, -2.84, z_pos), "size": vec(1.56, walls_thickness, walls_width)},
                {"pos": vec(-6.18, -8.97, z_pos), "size": vec(walls_thickness, 4.51, walls_width)},
                {"pos": vec(-0.64, -9.27, z_pos), "size": vec(11.61, walls_thickness, walls_width)},
                {"pos": vec(0.13, -8.45, z_pos), "size": vec(walls_thickness, 2.15, walls_width)},
                {"pos": vec(4.93, -3.83, z_pos), "size": vec(walls_thickness, 11.42, walls_width)},

                {"pos": vec(3.95, -0.33, z_pos), "size": vec(2.51, walls_thickness, walls_width)},
                {"pos": vec(7.03, 1.61, z_pos), "size": vec(4.62, walls_thickness, walls_width)},
                {"pos": vec(7.89, -4.29, z_pos), "size": vec(6.39, walls_thickness, walls_width)},
                {"pos": vec(9.13, 0.35, z_pos), "size": vec(walls_thickness, 13.95, walls_width)},
                {"pos": vec(9.83, 4.06, z_pos), "size": vec(1.96, walls_thickness, walls_width)},

                {"pos": vec(9.09, 7.06, z_pos), "size": vec(2.88, walls_thickness, walls_width)},
                {"pos": vec(7.93, 8.05, z_pos), "size": vec(walls_thickness, 2.51, walls_width)},
                {"pos": vec(7.09, 9.05, z_pos), "size": vec(2.25, walls_thickness, walls_width)},
                {"pos": vec(12.89, 5.54, z_pos), "size": vec(1.41, walls_thickness, walls_width)},
                {"pos": vec(12.58, -0.86, z_pos), "size": vec(1.98, walls_thickness, walls_width)},

                {"pos": vec(7.77, -9.66, z_pos), "size": vec(walls_thickness, 3, walls_width)}
            ]

            # walls (field with 21 holes)
            """walls_data = [
                {"pos": vec(4.19, 9.38, z_pos), "size": vec(walls_thickness, 4, walls_width)},
                {"pos": vec(8.88, 9.79, z_pos), "size": vec(9.63, walls_thickness, walls_width)},
                {"pos": vec(-1.35, 9.44, z_pos), "size": vec(11.68, walls_thickness, walls_width)},
                {"pos": vec(-6.94, 7.63, z_pos), "size": vec(walls_thickness, 4.15, walls_width)},
                {"pos": vec(-6.51, 5.83, z_pos), "size": vec(1.42, walls_thickness, walls_width)},

                {"pos": vec(-9.41, 9.63, z_pos), "size": vec(walls_thickness, 3.38, walls_width)},
                {"pos": vec(-11.43, 8.21, z_pos), "size": vec(4.52, walls_thickness, walls_width)},
                {"pos": vec(-10.94, 3.02, z_pos), "size": vec(5.51, walls_thickness, walls_width)},
                {"pos": vec(-10.37, 4.45, z_pos), "size": vec(walls_thickness, 3.41, walls_width)},
                {"pos": vec(-10.94, 0.29, z_pos), "size": vec(5.51, walls_thickness, walls_width)},

                {"pos": vec(-5.56, 0.41, z_pos), "size": vec(walls_thickness, 5.46, walls_width)},
                {"pos": vec(-4.85, 0.57, z_pos), "size": vec(1.98, walls_thickness, walls_width)},
                {"pos": vec(-4.11, -2.04, z_pos), "size": vec(6.56, walls_thickness, walls_width)},
                {"pos": vec(-7.11, -3, z_pos), "size": vec(walls_thickness, 2.42, walls_width)},
                {"pos": vec(-1.03, -4.03, z_pos), "size": vec(walls_thickness, 4.52, walls_width)},

                {"pos": vec(-9.61, -4.13, z_pos), "size": vec(walls_thickness, 2.73, walls_width)},
                {"pos": vec(-10.36, -4, z_pos), "size": vec(2.06, walls_thickness, walls_width)},
                {"pos": vec(-9.26, -9.41, z_pos), "size": vec(walls_thickness, 3.95, walls_width)},
                {"pos": vec(-11.34, -8.57, z_pos), "size": vec(4.71, walls_thickness, walls_width)},
                {"pos": vec(-5.71, -7.33, z_pos), "size": vec(3.01, walls_thickness, walls_width)},

                {"pos": vec(-4.48, -8.17, z_pos), "size": vec(walls_thickness, 6.44, walls_width)},
                {"pos": vec(1.06, -8.58, z_pos), "size": vec(11.57, walls_thickness, walls_width)},
                {"pos": vec(4.92, -7.11, z_pos), "size": vec(walls_thickness, 3.47, walls_width)},
                {"pos": vec(2.41, 0.93, z_pos), "size": vec(walls_thickness, 9.37, walls_width)},
                {"pos": vec(3.13, -3.07, z_pos), "size": vec(1.99, walls_thickness, walls_width)},

                {"pos": vec(0.07, 5.35, z_pos), "size": vec(5.12, walls_thickness, walls_width)},
                {"pos": vec(-2.22, 4.89, z_pos), "size": vec(walls_thickness, 3.82, walls_width)},
                {"pos": vec(0.26, 1.33, z_pos), "size": vec(walls_thickness, 2.73, walls_width)},
                {"pos": vec(4.83, 4.3, z_pos), "size": vec(walls_thickness, 1.4, walls_width)},
                {"pos": vec(5.87, 4.72, z_pos), "size": vec(2.65, walls_thickness, walls_width)},

                {"pos": vec(6.95, 5.15, z_pos), "size": vec(walls_thickness, 1.4, walls_width)},
                {"pos": vec(9.12, 7.38, z_pos), "size": vec(walls_thickness, 1.33, walls_width)},
                {"pos": vec(10.11, 7.77, z_pos), "size": vec(2.53, walls_thickness, walls_width)},
                {"pos": vec(12.31, 3.93, z_pos), "size": vec(2.76, walls_thickness, walls_width)},
                {"pos": vec(10.58, 0.82, z_pos), "size": vec(5.93, walls_thickness, walls_width)},

                {"pos": vec(7.89, -0.88, z_pos), "size": vec(walls_thickness, 3.97, walls_width)},
                {"pos": vec(6.69, -0.52, z_pos), "size": vec(2.97, walls_thickness, walls_width)},
                {"pos": vec(9.43, -2.6, z_pos), "size": vec(3.63, walls_thickness, walls_width)},
                {"pos": vec(11.04, -3.55, z_pos), "size": vec(walls_thickness, 2.47, walls_width)},
                {"pos": vec(9.31, -8.91, z_pos), "size": vec(walls_thickness, 4.97, walls_width)},
            ]"""

        # Dimension
        self.walls_data = walls_data
        self.walls_thickness = walls_thickness
        self.walls_width = walls_width

# -----------------------------------------------------------------------------
# fieldholes
# -----------------------------------------------------------------------------

class Holes:

    def __init__(self, holes_data=None, holes_radius =0.75, holes_depth = 0.31):
        if holes_data is None:
            # holes (field with 2 holes)
            """holes_data = [
                {"pos": vec(-1.6, 5.31, 0.001), "axis": vec(0.0, 0.0, -holes_depth)},
                {"pos": vec(2.06, -0.39, 0.001), "axis": vec(0.0, 0.0, -holes_depth)},
            ]"""

            #holes (field with 8 holes)
            holes_data = [
                {"pos": vec(-10.52, 7.03, 0.001), "axis": vec(0.0, 0.0, -holes_depth)},
                {"pos": vec(-8.06, 1.28, 0.001), "axis": vec(0.0, 0.0, -holes_depth)},
                {"pos": vec(-10.47, -5.03, 0.001), "axis": vec(0.0, 0.0, -holes_depth)},
                {"pos": vec(3.75, -7.91, 0.001), "axis": vec(0.0, 0.0, -holes_depth)},
                {"pos": vec(1.41, -0.45, 0.001), "axis": vec(0.0, 0.0, -holes_depth)},

                {"pos": vec(3.73, 5.45, 0.001), "axis": vec(0.0, 0.0, -holes_depth)},
                {"pos": vec(10.5, 2.69, 0.001), "axis": vec(0.0, 0.0, -holes_depth)},
                {"pos": vec(10.57, -5.8, 0.001), "axis": vec(0.0, 0.0, -holes_depth)}
            ]

            # holes (field with 21 holes)
            """holes_data = [
                {"pos": vec(-10.71, 7.03, 0.001), "axis": vec(0.0, 0.0, -holes_depth)},
                {"pos": vec(-5.81, 4.45, 0.001), "axis": vec(0.0, 0.0, -holes_depth)},
                {"pos": vec(-8.14, 1.32, 0.001), "axis": vec(0.0, 0.0, -holes_depth)},
                {"pos": vec(-8.16, -2.81, 0.001), "axis": vec(0.0, 0.0, -holes_depth)},
                {"pos": vec(-10.70, -5.11, 0.001), "axis": vec(0.0, 0.0, -holes_depth)},

                {"pos": vec(-8.01, -7.92, 0.001), "axis": vec(0.0, 0.0, -holes_depth)},
                #hole 7 = hole 4
                {"pos": vec(-3.29, -4.64, 0.001), "axis": vec(0.0, 0.0, -holes_depth)},
                {"pos": vec(3.69, -4.64, 0.001), "axis": vec(0.0, 0.0, -holes_depth)},
                {"pos": vec(1.22, -0.47, 0.001), "axis": vec(0.0, 0.0, -holes_depth)},

                {"pos": vec(-1.01, 3.35, 0.001), "axis": vec(0.0, 0.0, -holes_depth)},
                #hole 12 = hole 2
                {"pos": vec(-5.66, 7, 0.001), "axis": vec(0.0, 0.0, -holes_depth)},
                {"pos": vec(1.34, 7.73, 0.001), "axis": vec(0.0, 0.0, -holes_depth)},
                {"pos": vec(3.71, 2.84, 0.001), "axis": vec(0.0, 0.0, -holes_depth)},

                {"pos": vec(8.13, 6.08, 0.001), "axis": vec(0.0, 0.0, -holes_depth)},
                #hole 17 = hole 16
                {"pos": vec(10.53, 2.76, 0.001), "axis": vec(0.0, 0.0, -holes_depth)},
                #hole 19 = hole 15
                #hole 20 = hole 9
                {"pos": vec(10.53, -5.83, 0.001), "axis": vec(0.0, 0.0, -holes_depth)},
            ]"""

        # Dimension
        self.holes_data = holes_data
        self.holes_radius = holes_radius


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
            Weight in [g]. The default is 9.0. NOT USED

        Returns
        -------
        None.

        """
        self.radius = radius
        self.weightGram = weightGram
