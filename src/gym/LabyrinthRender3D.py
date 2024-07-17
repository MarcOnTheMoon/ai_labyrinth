"""
3D rendering for a labyrinth OpenAI gym environment.

The graphical user interface is based on VPython ( https://vpython.org/ ).
The library uses the default browser to display images and animations. Install
VPython in Anaconda by the command 'conda install -c conda-forge vpython'.

@authors: Marc Hensel, Sandra Lassahn
@contact: http://www.haw-hamburg.de/marc-hensel
@copyright: 2024
@version: 2024.05.14
@license: CC BY-NC-SA 4.0, see https://creativecommons.org/licenses/by-nc-sa/4.0/deed.en
"""
from vpython import scene, box, cylinder, sphere, rotate, compound, textures
from vpython import vector as vec
from math import pi
import time
from LabyrinthGeometry import LabyrinthGeometry

class LabyrinthRender3D:

    # ========== Colors =======================================================

    # Define colors for rendering
    colors = {
        'background': vec(0.6, 0.6, 0.6),
        'box'       : vec(0.9, 0.0, 0.0),
        'wheels'    : vec(0.2, 0.2, 0.2),
        'field'     : vec(0.9, 0.8, 0.6),
        'topfield'  : vec(1.0, 1.0, 1.0),
        'walls'     : vec(0.2, 0.2, 0.2),
        'holes'     : vec(0.1, 0.1, 0.1),
        'on_field'  : vec(1.0, 0.8, 0.6),
        'arrow_x'   : vec(0.0, 0.0, 1.0),
        'arrow_y'   : vec(1.0, 0.0, 0.0)
    }

    # ========== Constructor ==================================================

    def __init__(self, geometry, scene_width=800, scene_height=600, scene_range=20, ball_position=None):
        """
        Constructor.

        Parameters
        ----------
        geometry : LabyrinthGeometry
            Geometry to render.
        scene_width : int, optional
            Width of the display area in pixel. The default is 800.
        scene_height : int, optional
            Height of the display area in pixel. The default is 600.
        scene_range : float, optional
            Virtual distance of the camera viewing the scene. The default is 20.
        ball_position: vec(x, y, z)
            sets the starting position of the ball

        Returns
        -------
        None.

        """
        # Set VPython scene with dimensions
        self.__scene = scene
        self.__scene.width = scene_width
        self.__scene.height = scene_height
        self.__scene.range = scene_range
        self.__scene.center = vec(0, 0, -geometry.box.height/2)

        # Set further scene properties
        self.__scene.title = 'Labyrinth gym (right-click to rotate, mouse wheel to zoom)'
        self.__scene.caption = "\n"
        self.__scene.autoscale = False
        self.__scene.background = self.colors['background']

        # Set geometry and render objects
        self.__ball_radius = geometry.ball.radius
        if ball_position != None:
            self.__ball_position = ball_position
        else:
            self.__ball_position = vec(0,0,0)
        self.__x_degree = 0.0
        self.__y_degree = 0.0
        self.__axis_x = vec(0, -1, 0)
        self.__axis_y = vec(1, 0, 0)
        self.__render_scene(geometry)
        
    # ========== Initially render objects =====================================

    def __render_scene(self, geometry):
        """
        Render all objects in the scene.

        Parameters
        ----------
        geometry : LabyrinthGeometry
            Labyrinth layout to render.

        Returns
        -------
        None.

        """
        # Render static objects
        self.__render_box(geometry)
        self.__render_field(geometry)

        # Render ball at start location
        self.__render_ball(self.__ball_position.x, self.__ball_position.y)

    # ========== Render static objects ========================================

    def __render_box(self, geometry):
        """
        Render the box housing the tiltable field.

        Parameters
        ----------
        geometry : LabyrinthGeometry
            Labyrinth layout to render.

        Returns
        -------
        None.

        """
        # Init geometric values
        field_x = geometry.field.size_x
        field_y = geometry.field.size_y
        height = geometry.box.height
        thickness = geometry.box.boarder
        wheel_radius = geometry.box.wheel_radius
        wheel_depth = geometry.box.wheel_depth
        dz = 1.0 #boarder height (Randhöhe)

        # Ground plate
        center = vec(0, 0, -(height - thickness/2) + dz)
        box(pos=center, length=field_x, height=field_y, width=thickness, color=self.colors['field'])

        # Upper and lower plate
        upper = vec(0,  (field_y + thickness)/2, -height/2 + dz)
        lower = vec(0, -(field_y + thickness)/2, -height/2 + dz)
        box(pos=upper, length=field_x + 2 * thickness, height=thickness, width=height, color=self.colors['box'])
        box(pos=lower, length=field_x + 2 * thickness, height=thickness, width=height, color=self.colors['box'])

        # Left and right plate        
        left  = vec(-(field_x + thickness)/2, 0, -height/2 + dz)
        right = vec( (field_x + thickness)/2, 0, -height/2 + dz)
        box(pos=left,  length=thickness, height=field_y, width=height, color=self.colors['box'])
        box(pos=right, length=thickness, height=field_y, width=height, color=self.colors['box'])        
        
        # Wheels
        front = vec(0, -(field_y/2 + thickness), -height/2 + dz)
        right = vec((field_x/2 + thickness), 0, -height/2 + dz)
        cylinder(pos=front, axis=vec(0, -1, 0), radius=wheel_radius, length=wheel_depth, color=self.colors['wheels'])
        cylinder(pos=right, axis=vec(1,  0, 0), radius=wheel_radius, length=wheel_depth, color=self.colors['wheels'])

    # ========== Render dynamic objects =======================================

    def __render_field(self, geometry):
        """
        Render playing field.

        Parameters
        ----------
        geometry : LabyrinthGeometry
            Labyrinth layout to render.

        Returns
        -------
        None.

        """

        # Init geometric values
        field_x = geometry.field.size_x
        field_y = geometry.field.size_y
        plate_depth = geometry.field.plate_depth

        # Render plate (Coordinate system's origin on top surface of plate)
        center = vec(0, 0, -plate_depth/2)
        field = box(pos=center, length=field_x, height=field_y, width=plate_depth, color=self.colors['topfield'])

        # Init list of elements
        labyrinth_elements = []
        labyrinth_elements.append(field)

        if geometry.layout != '0 holes' and geometry.layout != '0 holes real':
            # Add walls
            walls_data = geometry.walls.data
            for wall_data in walls_data:
                wall = box(pos=wall_data["pos"], size=wall_data["size"], color=self.colors['walls'])
                labyrinth_elements.append(wall)

            # Add holes
            holes_data = geometry.holes.data
            holes_radius = geometry.holes.radius
            for hole_data in holes_data:
                hole = cylinder(pos=hole_data["pos"], axis=hole_data["axis"], radius=holes_radius, color=self.colors['holes'])
                labyrinth_elements.append(hole)

        # Compile all elements into a board with background picture
        if geometry.layout == '0 holes':
            labyrinth = compound(labyrinth_elements, texture='textures/0_holes.png')
        elif geometry.layout == '0 holes real':
            labyrinth = compound(labyrinth_elements, texture='textures/0_holes_real.png')
        elif geometry.layout == '2 holes':
            labyrinth = compound(labyrinth_elements, texture='textures/2_holes.png')
        elif geometry.layout == '2 holes real':
            labyrinth = compound(labyrinth_elements, texture='textures/2_holes_real.png')
        elif geometry.layout == '8 holes':
            labyrinth = compound(labyrinth_elements, texture='textures/8_holes.png')
        elif geometry.layout == '21 holes':
            labyrinth = compound(labyrinth_elements, texture='textures/21_holes.png')
        self.__labyrinth = labyrinth

        # Rotate plate
        self.rotate_by(x_degree=geometry.field.rotation_x_deg, y_degree=geometry.field.rotation_y_deg)

    # -------------------------------------------------------------------------

    def __render_ball(self, x, y):
        """
        Render the ball.

        The (x,y)-coordinate are relative to the center of the playing field.
        The z-coordinate is set to the radius, so that the ball is placed
        on the field.

        Parameters
        ----------
        x : float
            x-coordinate of ball's center.
        y : float
            y-coordinate of ball's center.

        Returns
        -------
        None.

        """
        if self.__ball_position == None:
            position = vec(x, y, self.__ball_radius)
        else:
            position = self.__ball_position
            position.z = self.__ball_radius
        self.__ball = sphere(pos=position, radius=self.__ball_radius, texture=textures.metal)

    def ball_visibility(self, visible):
        self.__ball.visible = visible

    # ========== Move dynamic objects ========================================

    def rotate_by(self, x_degree=None, y_degree=None):
        """
        Rotate the field by the angles passed as parameters.

        To keep all sides parallel to the outer box, rotations of the physical
        game and this environment are done as follows:

        - Left/right with fixed rotation axis along y-direction (0, -1, 0)
        - Front/back around axis (1, 0, dz) tilting with left/right rotation

        Parameters
        ----------
        x_degree : float, optional
            Rotation angle in x-direction [degree]. The default is None.
        y_degree : float, optional
            Rotation angle in y-direction [degree]. The default is None.

        Returns
        -------
        None.

        """
        if x_degree != None:
            x_rad = x_degree * pi/180.
            self.__x_degree = self.__x_degree + x_degree
            self.__labyrinth.rotate(angle=x_rad, axis=self.__axis_x, origin=vec(0, 0, 0))
            self.__axis_y = rotate(self.__axis_y, angle=x_rad, axis=self.__axis_x)

        if y_degree != None:
            y_rad = y_degree * pi/180.
            self.__y_degree = self.__y_degree + y_degree
            self.__labyrinth.rotate(angle=y_rad, axis=self.__axis_y, origin=vec(0, 0, 0))

    # -------------------------------------------------------------------------

    def rotate_to(self, x_degree=None, y_degree=None):
        """
        Rotate the field to the angles passed as parameters.

        Parameters
        ----------
        x_degree : float, optional
            Target angle in x-direction [degree]. The default is None.
        y_degree : float, optional
            Target angle in y-direction [degree]. The default is None.

        Returns
        -------
        None.

        """
        if x_degree != None:
            delta_x = x_degree - self.__x_degree
        if y_degree != None:
            delta_y = y_degree - self.__y_degree

        self.rotate_by(x_degree=delta_x, y_degree=delta_y)

    def get_x_rad(self):
        return self.__x_degree * pi/180.0

    def get_y_rad(self):
        return self.__y_degree * pi/180.0

    # -------------------------------------------------------------------------

    def move_ball(self, x, y):
        """
        Move the ball to a specific location.

        Parameters
        ----------
        x : float
            x-coordinate of ball's center.
        y : float
            y-coordinate of ball's center.

        Returns
        -------
        None.

        """
        self.__ball.pos = vec(x, y, self.__ball_radius)

        self.__ball.rotate(angle=self.get_x_rad(), axis=self.__axis_x, origin=vec(0,0,0))
        self.__ball.rotate(angle=self.get_y_rad(), axis=self.__axis_y, origin=vec(0,0,0))

# -----------------------------------------------------------------------------
# Main (sample)
# -----------------------------------------------------------------------------

if __name__ == '__main__':
    # Render initial pinball environment
    geometry = LabyrinthGeometry(layout='8 holes')
    render = LabyrinthRender3D(geometry)
    time.sleep(2.0)

    # Do some sample movements
    number_steps = 25
    alpha = 0.25

    for i in range(number_steps):
        render.rotate_by(x_degree=alpha)
        time.sleep(0.05)
    for i in range(2 * number_steps):
        render.rotate_by(x_degree=-alpha)
        time.sleep(0.05)
    for i in range(number_steps):
        render.rotate_by(x_degree=alpha, y_degree=alpha)
        time.sleep(0.05)
    for i in range(2 * number_steps):
        render.rotate_by(y_degree=-alpha)
        time.sleep(0.05)
    for i in range(number_steps):
        render.rotate_by(y_degree=alpha)
        time.sleep(0.01)