"""
Layouts with geometric objects for a labyrinth OpenAI gym environment.

Lengths are stated without unit, but are interpreted as [cm].

@authors: Marc Hensel, Sandra Lassahn
@contact: http://www.haw-hamburg.de/marc-hensel
@copyright: 2024
@version: 2024.08.29
@license: CC BY-NC-SA 4.0, see https://creativecommons.org/licenses/by-nc-sa/4.0/deed.en
"""
import random
import numpy as np
from enum import Enum
from vpython import vector as vec

# -----------------------------------------------------------------------------
# Enumeration of field layouts
# -----------------------------------------------------------------------------

class Layout(Enum):
    """
    Enumerated constants representing labyrinth field layouts.
    
    Layouts without the suffix 'VIRTUAL' exist identically for the virtual
    environment and the physical labyrinth device.
    
    Layouts with the suffix 'VIRTUAL' do not exist for the physical device,
    however, they are kept because trained data is available for these layouts
    in the virtual environment.

    Enumeration elements must differ in terms of parameters; otherwise,
    they will be mapped to the same value, and not all game boards can be used.
    """
    HOLES_0_VIRTUAL = (0, True)
    HOLES_2_VIRTUAL = (2, True)
    HOLES_0  = (0, False)
    HOLES_2  = (2, False)
    HOLES_8  = (8, False)
    HOLES_21 = (21, False)
    
    # ========== Constructor ==================================================
    
    def __init__(self, number_holes, virtual):
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
    
# -----------------------------------------------------------------------------
# Class containing individual components of labyrinth game geometry
# -----------------------------------------------------------------------------

class Geometry:
    """
    Representation of complete labyrinth geometry.
    """

    # ========== Settings =====================================================
    # TODO Replace all these 'settings' by parameters and commented out code

    # Set start areas for random start position as [x_min, x_max, y_min, y_max]
    start_area = np.array([-6.06, 6.06, -5.76, 5.76], dtype=np.float32)         # Layout.HOLES_0_VIRTUAL, close to destination
    
    # Set random set for random start position
    random.seed(1)
    
    # ========== Field layouts ================================================

    # Define start positions and destination areas
    start_positions = {
        Layout.HOLES_0_VIRTUAL  : np.array([random.uniform(start_area[0], start_area[1]), random.uniform(start_area[2], start_area[3]), 0], dtype=np.float32),
        Layout.HOLES_0          : np.array([random.uniform(start_area[0], start_area[1]), random.uniform(start_area[2], start_area[3]), 0], dtype=np.float32),
        Layout.HOLES_2_VIRTUAL  : np.array([-1.52, 9.25, 0], dtype=np.float32),
        Layout.HOLES_2          : np.array([-0.79, 9.86, 0], dtype=np.float32),
        Layout.HOLES_8          : np.array([0.13, 10.53, 0], dtype=np.float32),
        #Layout.HOLES_8         : np.array([13, -5.0, 0], dtype=np.float32)     # Closer to the destination position
        Layout.HOLES_21         : np.array([3.2, 10.47, 0], dtype=np.float32)
        }
    destinations_xy = {
        Layout.HOLES_0_VIRTUAL  : [[-0.25, 0.25], [-0.25, 0.25]],
        Layout.HOLES_0          : [[-0.25, 0.25], [-0.25, 0.25]],
        Layout.HOLES_2_VIRTUAL  : [[-1.9, 1.56], [-6.62, -5.86]],
        Layout.HOLES_2          : [[-0.24, 2.73], [-11.4, -10.08]],
        Layout.HOLES_8          : [[-5.9, -4.33], [-11.4, -9.52]],
        #Layout.HOLES_8         : [[0.16, 4.59], [-6.95, -3.85]],   # Closer to the start position
        Layout.HOLES_21         : [[-4.2, -3.3], [-11.4, -8.91]]
        }

    # ========== Constructor ==================================================

    def __init__(self, layout):
        """
        Constructor.

        Parameters
        ----------
        layout : enum Layout
            Layout of holes and walls

        Returns
        -------
        None.

        """
        # Set layout information
        assert type(layout) == Layout
        self.layout = layout
        
        # Init geometry
        self.box = Box()
        self.field = Field()
        if layout.number_holes != 0:
            self.walls = Walls(layout=layout)
            self.holes = Holes(layout=layout, depth=self.field.plate_depth)
        self.ball = Ball()

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
# Walls on field defining track
# -----------------------------------------------------------------------------

class Walls:
    """
    Representation of walls defining the ball's track.
    """
    
    def __init__(self, layout, thickness=0.58, height=0.6):
        """
        Constructor.

        Parameters
        ----------
        layout : enum Layout
            Layout to generate.
        thickness : float, optional
            Thickness of walls. The default is 0.58.
        height : float, optional
            Height of walls above the playing field. The default is 0.6.

        Returns
        -------
        None.

        """
        # Dimension
        self.thickness = thickness
        self.height = height

        # Data
        z_pos = height / 2
        match layout:
            case Layout.HOLES_2_VIRTUAL:
                self.data = [
                    {"pos": vec(-3.41, 3.5, z_pos), "size": vec(thickness, 15.81, height)},
                    {"pos": vec(-2.37, 4, z_pos), "size": vec(2.64, thickness, height)},
                    {"pos": vec(-7.78, -4.54, z_pos), "size": vec(11.83, thickness, height)},
                    {"pos": vec(-2.14, -7.82, z_pos), "size": vec(thickness, 7.12, height)},
                    {"pos": vec(-0.13, -6.9, z_pos), "size": vec(4.45, thickness, height)},
    
                    {"pos": vec(1.84, -7.82, z_pos), "size": vec(thickness, 7.12, height)},
                    {"pos": vec(7.53, -4.54, z_pos), "size": vec(11.95, thickness, height)},
                    {"pos": vec(3.61, 3.5, z_pos), "size": vec(thickness, 15.81, height)}
                ]

            case Layout.HOLES_2:
                self.data = [
                    {"pos": vec(-2.44, 1.08, z_pos), "size": vec(thickness, 20.65, height)},
                    {"pos": vec(-1.35, 2.35, z_pos), "size": vec(2.75, thickness, height)},
                    {"pos": vec(-1.54, -8.97, z_pos), "size": vec(2.34, thickness, height)},
                    {"pos": vec(-0.65, -10.12, z_pos), "size": vec(thickness, 2.56, height)},
                    {"pos": vec(4.78, 1.08, z_pos), "size": vec(thickness, 20.65, height)},
    
                    {"pos": vec(3.91, -8.97, z_pos), "size": vec(2.31, thickness, height)},
                    {"pos": vec(3.04, -10.12, z_pos), "size": vec(thickness, 2.56, height)},
                ]

            case Layout.HOLES_8:
                self.data = [
                    {"pos": vec(2.01, 9.09, z_pos), "size": vec(thickness, 4.54, height)},
                    {"pos": vec(-4.08, 9.49, z_pos), "size": vec(11.72, thickness, height)},
                    {"pos": vec(-9.66, 9.09, z_pos), "size": vec(thickness, 1.42, height)},
                    {"pos": vec(2.82, 7.1, z_pos), "size": vec(2.04, thickness, height)},
                    {"pos": vec(-11.57, 5.65, z_pos), "size": vec(thickness, 2.36, height)},
    
                    {"pos": vec(-6.92, 5.85, z_pos), "size": vec(9.85, thickness, height)},
                    {"pos": vec(-2.28, 0.96, z_pos), "size": vec(thickness, 10.32, height)},
                    {"pos": vec(0.44, 4, z_pos), "size": vec(5.92, thickness, height)},
                    {"pos": vec(-3.62, -3.96, z_pos), "size": vec(11.17, thickness, height)},
                    {"pos": vec(-3.32, -5.18, z_pos), "size": vec(thickness, 3.01, height)},
    
                    {"pos": vec(-8.96, -4.85, z_pos), "size": vec(thickness, 3.92, height)},
                    {"pos": vec(-9.82, -6.55, z_pos), "size": vec(2.28, thickness, height)},
                    {"pos": vec(-9.19, 0.11, z_pos), "size": vec(8.73, thickness, height)},
                    {"pos": vec(-10.11, 0.91, z_pos), "size": vec(thickness, 2.13, height)},
                    {"pos": vec(-6.51, 0.91, z_pos), "size": vec(thickness, 2.13, height)},
    
                    {"pos": vec(-12.83, -2.84, z_pos), "size": vec(1.56, thickness, height)},
                    {"pos": vec(-6.18, -8.97, z_pos), "size": vec(thickness, 4.51, height)},
                    {"pos": vec(-0.64, -9.27, z_pos), "size": vec(11.61, thickness, height)},
                    {"pos": vec(0.13, -8.45, z_pos), "size": vec(thickness, 2.15, height)},
                    {"pos": vec(4.93, -3.83, z_pos), "size": vec(thickness, 11.42, height)},
    
                    {"pos": vec(3.95, -0.33, z_pos), "size": vec(2.51, thickness, height)},
                    {"pos": vec(7.03, 1.61, z_pos), "size": vec(4.62, thickness, height)},
                    {"pos": vec(7.89, -4.29, z_pos), "size": vec(6.39, thickness, height)},
                    {"pos": vec(9.13, 0.35, z_pos), "size": vec(thickness, 13.95, height)},
                    {"pos": vec(9.83, 4.06, z_pos), "size": vec(1.96, thickness, height)},
    
                    {"pos": vec(9.09, 7.06, z_pos), "size": vec(2.88, thickness, height)},
                    {"pos": vec(7.93, 8.05, z_pos), "size": vec(thickness, 2.51, height)},
                    {"pos": vec(7.09, 9.05, z_pos), "size": vec(2.25, thickness, height)},
                    {"pos": vec(12.89, 5.54, z_pos), "size": vec(1.41, thickness, height)},
                    {"pos": vec(12.58, -0.86, z_pos), "size": vec(1.98, thickness, height)},
    
                    {"pos": vec(7.77, -9.66, z_pos), "size": vec(thickness, 3, height)}
                ]

            case Layout.HOLES_21:
                self.data = [
                    {"pos": vec(4.19, 9.38, z_pos), "size": vec(thickness, 4, height)},
                    {"pos": vec(8.88, 9.79, z_pos), "size": vec(9.63, thickness, height)},
                    {"pos": vec(-1.35, 9.44, z_pos), "size": vec(11.68, thickness, height)},
                    {"pos": vec(-6.94, 7.63, z_pos), "size": vec(thickness, 4.15, height)},
                    {"pos": vec(-6.51, 5.83, z_pos), "size": vec(1.42, thickness, height)},
    
                    {"pos": vec(-9.41, 9.63, z_pos), "size": vec(thickness, 3.38, height)},
                    {"pos": vec(-11.43, 8.21, z_pos), "size": vec(4.52, thickness, height)},
                    {"pos": vec(-10.94, 3.02, z_pos), "size": vec(5.51, thickness, height)},
                    {"pos": vec(-10.37, 4.45, z_pos), "size": vec(thickness, 3.41, height)},
                    {"pos": vec(-10.94, 0.29, z_pos), "size": vec(5.51, thickness, height)},
    
                    {"pos": vec(-5.56, 0.41, z_pos), "size": vec(thickness, 5.46, height)},
                    {"pos": vec(-4.85, 0.57, z_pos), "size": vec(1.98, thickness, height)},
                    {"pos": vec(-4.11, -2.04, z_pos), "size": vec(6.56, thickness, height)},
                    {"pos": vec(-7.11, -3, z_pos), "size": vec(thickness, 2.42, height)},
                    {"pos": vec(-1.03, -4.03, z_pos), "size": vec(thickness, 4.52, height)},
    
                    {"pos": vec(-9.61, -4.13, z_pos), "size": vec(thickness, 2.73, height)},
                    {"pos": vec(-10.36, -4, z_pos), "size": vec(2.06, thickness, height)},
                    {"pos": vec(-9.26, -9.41, z_pos), "size": vec(thickness, 3.95, height)},
                    {"pos": vec(-11.34, -8.57, z_pos), "size": vec(4.71, thickness, height)},
                    {"pos": vec(-5.71, -7.33, z_pos), "size": vec(3.01, thickness, height)},
    
                    {"pos": vec(-4.48, -8.17, z_pos), "size": vec(thickness, 6.44, height)},
                    {"pos": vec(1.06, -8.58, z_pos), "size": vec(11.57, thickness, height)},
                    {"pos": vec(4.92, -7.11, z_pos), "size": vec(thickness, 3.47, height)},
                    {"pos": vec(2.41, 0.93, z_pos), "size": vec(thickness, 9.37, height)},
                    {"pos": vec(3.13, -3.07, z_pos), "size": vec(1.99, thickness, height)},
    
                    {"pos": vec(0.07, 5.35, z_pos), "size": vec(5.12, thickness, height)},
                    {"pos": vec(-2.22, 4.89, z_pos), "size": vec(thickness, 3.82, height)},
                    {"pos": vec(0.26, 1.33, z_pos), "size": vec(thickness, 2.73, height)},
                    {"pos": vec(4.83, 4.3, z_pos), "size": vec(thickness, 1.4, height)},
                    {"pos": vec(5.87, 4.72, z_pos), "size": vec(2.65, thickness, height)},
    
                    {"pos": vec(6.95, 5.15, z_pos), "size": vec(thickness, 1.4, height)},
                    {"pos": vec(9.12, 7.38, z_pos), "size": vec(thickness, 1.33, height)},
                    {"pos": vec(10.11, 7.77, z_pos), "size": vec(2.53, thickness, height)},
                    {"pos": vec(12.31, 3.93, z_pos), "size": vec(2.76, thickness, height)},
                    {"pos": vec(10.58, 0.82, z_pos), "size": vec(5.93, thickness, height)},
    
                    {"pos": vec(7.89, -0.88, z_pos), "size": vec(thickness, 3.97, height)},
                    {"pos": vec(6.69, -0.52, z_pos), "size": vec(2.97, thickness, height)},
                    {"pos": vec(9.43, -2.6, z_pos), "size": vec(3.63, thickness, height)},
                    {"pos": vec(11.04, -3.55, z_pos), "size": vec(thickness, 2.47, height)},
                    {"pos": vec(9.31, -8.91, z_pos), "size": vec(thickness, 4.97, height)},
                ]
            
            case _:
                self.data = None

# -----------------------------------------------------------------------------
# Holes in field
# -----------------------------------------------------------------------------

class Holes:
    """
    Representation of holes in the field.
    
    These are the holes that balls can fall into, which ends the game.
    """
    
    def __init__(self, layout, radius=0.75, depth=0.3):
        """
        Constructor.

        Parameters
        ----------
        layout : enum Layout
            Layout to generate.
        radius : float, optional
            Radius of the holes. The default is 0.75.
        depth : float, optional
            Depth of the holes. The default is 0.3.

        Returns
        -------
        None.

        """
        # Dimension
        self.radius = radius
        match layout:
            case Layout.HOLES_2_VIRTUAL:
                self.data = [
                    {"pos": vec(-1.6, 5.31, 0.001), "axis": vec(0.0, 0.0, -depth)},
                    {"pos": vec(2.06, -0.39, 0.001), "axis": vec(0.0, 0.0, -depth)},
                ]
    
            case Layout.HOLES_2:
                self.data = [
                    {"pos": vec(-1.12, 3.36, 0.001), "axis": vec(0.0, 0.0, -depth)},
                    {"pos": vec(3.41, -4.72, 0.001), "axis": vec(0.0, 0.0, -depth)},
                ]
    
            case Layout.HOLES_8:
                self.data = [
                    {"pos": vec(-10.52, 7.03, 0.001), "axis": vec(0.0, 0.0, -depth)},
                    {"pos": vec(-8.06, 1.28, 0.001), "axis": vec(0.0, 0.0, -depth)},
                    {"pos": vec(-10.47, -5.03, 0.001), "axis": vec(0.0, 0.0, -depth)},
                    {"pos": vec(3.75, -7.91, 0.001), "axis": vec(0.0, 0.0, -depth)},
                    {"pos": vec(1.41, -0.45, 0.001), "axis": vec(0.0, 0.0, -depth)},
    
                    {"pos": vec(3.73, 5.45, 0.001), "axis": vec(0.0, 0.0, -depth)},
                    {"pos": vec(10.5, 2.69, 0.001), "axis": vec(0.0, 0.0, -depth)},
                    {"pos": vec(10.57, -5.8, 0.001), "axis": vec(0.0, 0.0, -depth)}
                ]

            case Layout.HOLES_21:
                self.data = [
                    {"pos": vec(-10.71, 7.03, 0.001), "axis": vec(0.0, 0.0, -depth)},
                    {"pos": vec(-5.81, 4.45, 0.001), "axis": vec(0.0, 0.0, -depth)},
                    {"pos": vec(-8.14, 1.32, 0.001), "axis": vec(0.0, 0.0, -depth)},
                    {"pos": vec(-8.16, -2.81, 0.001), "axis": vec(0.0, 0.0, -depth)},
                    {"pos": vec(-10.70, -5.11, 0.001), "axis": vec(0.0, 0.0, -depth)},
    
                    {"pos": vec(-8.01, -7.92, 0.001), "axis": vec(0.0, 0.0, -depth)},
                    #hole 7 = hole 4
                    {"pos": vec(-3.29, -4.64, 0.001), "axis": vec(0.0, 0.0, -depth)},
                    {"pos": vec(3.69, -4.64, 0.001), "axis": vec(0.0, 0.0, -depth)},
                    {"pos": vec(1.22, -0.47, 0.001), "axis": vec(0.0, 0.0, -depth)},
    
                    {"pos": vec(-1.01, 3.35, 0.001), "axis": vec(0.0, 0.0, -depth)},
                    #hole 12 = hole 2
                    {"pos": vec(-5.66, 7, 0.001), "axis": vec(0.0, 0.0, -depth)},
                    {"pos": vec(1.34, 7.73, 0.001), "axis": vec(0.0, 0.0, -depth)},
                    {"pos": vec(3.71, 2.84, 0.001), "axis": vec(0.0, 0.0, -depth)},
    
                    {"pos": vec(8.13, 6.08, 0.001), "axis": vec(0.0, 0.0, -depth)},
                    #hole 17 = hole 16
                    {"pos": vec(10.53, 2.76, 0.001), "axis": vec(0.0, 0.0, -depth)},
                    #hole 19 = hole 15
                    #hole 20 = hole 9
                    {"pos": vec(10.53, -5.83, 0.001), "axis": vec(0.0, 0.0, -depth)},
                ]
        
            case _:
                self.data = None

# -----------------------------------------------------------------------------
# Ball
# -----------------------------------------------------------------------------

class Ball:
    """
    Representation of ball.
    
    The default values correspond to a ball from the game series Labyrinth and
    GraviTrax, having a diameter of 12.7 mm.
    """
    
    def __init__(self, radius=0.635):
        """
        Constructor.

        Parameters
        ----------
        radius : float, optional
            Raidus in [cm]. The default is 0.635.

        Returns
        -------
        None.

        """
        self.radius = radius
