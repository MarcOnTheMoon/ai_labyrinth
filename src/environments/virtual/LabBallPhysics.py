"""
Models realistic behavior of the ball on the field for a labyrinth OpenAI gym environment.

@authors: Sandra Lassahn, Marc Hensel
@contact: http://www.haw-hamburg.de/marc-hensel
@copyright: 2024
@version: 2024.08.29
@license: CC BY-NC-SA 4.0, see https://creativecommons.org/licenses/by-nc-sa/4.0/deed.en
"""
from vpython import vector as vec
from math import sqrt, sin, cos
import time
import numpy as np
from LabRender3D import Render3D
from LabLayouts import Layout, Geometry

# TODO Corners: vpython vector to numpy array for better performance. (Walls only used once to init corners.)

class BallPhysics:

    # ========== Constructor ==================================================

    def __init__(self, geometry, dt):
        """
        Constructor.

        Parameters
        ----------
        geometry : Geometry
            Geometry for calculations
        dt : float
            Time period for each step [s]

        Returns
        -------
        None.

        """
        # Timing
        self.dt = dt                    # Time step [s]

        # Ball state
        self.is_in_hole = False                                     # Ball has fallen into hole if True
        self.__position = np.array([0.0, 0.0], dtype=np.float32)    # Ball's position [cm]^3
        self.__velocity = np.array([0.0, 0.0], dtype=np.float32)    # Ball's velocity [cm/s]^3
        
        # Field tilting
        self.__x_rad = 0.0
        self.__y_rad = 0.0
        
        # Physical constants
        g = 9.81 * 100.0                # Acceleration of gravity [cm/s²]
        self.__5_7g = 5.0 / 7.0 * g     # Constant for ball acceleration [cm/s²]
        self.__urr = 0.00118            # Rolling friction coefficient
        self.__urr_edge = 0.00212       # Friction coefficients for collision with an edge

        # Geometry        
        self.__geometry = geometry
        self.__init_corners()

    # ========== Public methods ===============================================
    
    def reset(self, position):
        """
        Reset ball physics.

        Parameters
        ----------
        position : numpy.float32[2]
            Position of ball.

        Returns
        -------
        None.

        """
        # Ball
        self.__position = position
        self.__velocity = np.array([0.0, 0.0], dtype=np.float32)
        self.is_in_hole = False
        
        # Field tilting
        self.__x_rad = 0.0
        self.__y_rad = 0.0

    # -------------------------------------------------------------------------
    
    def step(self, x_rad, y_rad, number_steps=1, position=None):
        """
        Simulates ball's position and velocity after one or more time steps dt.

        Parameters
        ----------
        x_rad : float
            Field's tilting angle in x-direction after the time steps [rad].
        y_rad : float
            Field's tilting angle in y-direction after the time steps [rad].
        number_steps : int, optional
            Number of time steps of duration dt to perform. The default is 1.
        position : numpy.float32[2], optional
            Start position of the ball. The default is None.

        Returns
        -------
        position : numpy.float32[2]
            New ball position

        """
        # Set ball's position
        if position != None:
            self.set_position(position)

        # Set field rotation
        self.__x_rad = x_rad
        self.__y_rad = y_rad

        # Do time steps    
        for i in range(number_steps):
            self.__step(x_rad=x_rad, y_rad=y_rad)
        
        return self.__position

    # ========== Initialize labyrinth geometry ================================
    
    def __init_corners(self):
        """
        Calculates all corner points of the individual labyrinth walls.

        Parameters
        ----------
        None.

        Returns
        -------
        None.

        """
        if self.__geometry.layout.number_holes > 0:
            # Interior walls (copied from Geometry)
            walls_data = self.__geometry.walls.data.copy()
        else:
            walls_data = []
        
        # Add exterior walls (field boarders)
        field_x = self.__geometry.field.size_x
        field_y = self.__geometry.field.size_y
        height = self.__geometry.box.height
        thickness = self.__geometry.box.boarder
        
        walls_data.append({"pos": vec(0,  (field_y + thickness)/2, 0), "size": vec(field_x + 2 * thickness, thickness, height)})    # Upper
        walls_data.append({"pos": vec(0, -(field_y + thickness)/2, 0), "size": vec(field_x + 2 * thickness, thickness, height)})    # Lower
        walls_data.append({"pos": vec(-(field_x + thickness)/2, 0, 0), "size": vec(thickness, field_y, height)})                    # Left
        walls_data.append({"pos": vec((field_x + thickness) / 2, 0, 0), "size": vec(thickness, field_y, height)})                   # Right

        # Corner calculation
        self.__corners = []
        for wall_data in walls_data:
            wall_center = wall_data["pos"]
            wall_size = wall_data["size"]
            corner_left_upper = vec( wall_center.x - wall_size.x /2, wall_center.y + wall_size.y /2, wall_center.z)
            corner_right_upper = vec(wall_center.x + wall_size.x / 2, wall_center.y + wall_size.y / 2, wall_center.z)
            corner_right_lower = vec(wall_center.x + wall_size.x / 2, wall_center.y - wall_size.y / 2, wall_center.z)
            corner_left_lower = vec(wall_center.x - wall_size.x / 2, wall_center.y - wall_size.y / 2, wall_center.z)

            self.__corners.append([corner_left_upper, corner_right_upper, corner_right_lower, corner_left_lower])

    # ========== Simulate one time step of duration dt ========================
    
    def __step(self, x_rad, y_rad):
        """
        Simulates ball's position and velocity after one time steps dt.
        
        The method checks and processes collisions with walls and corners and
        whether the ball falls into a hole.

        Parameters
        ----------
        x_rad : float
            Field's tilting angle in x-direction [rad].
        y_rad : float
            Field's tilting angle in y-direction [rad].

        Returns
        -------
        position : numpy.float32[2]
            New ball position

        """
        # Set arguments
        self.__x_rad = x_rad
        self.__y_rad = y_rad

        # ---------- x-direction ----------------------------------------------

        x_sin = sin(self.__x_rad)
        x_friction_coefficient = self.__urr * cos(self.__x_rad)

        # Acceleration (sign of friction coefficient depending on direction of speed in x)
        if self.__velocity[0] <= 0:
            x_a = -self.__5_7g * (x_sin - x_friction_coefficient)
        else:
            x_a = -self.__5_7g * (x_sin + x_friction_coefficient)

        # New velocity in x
        velocity_x = self.__velocity[0] + x_a * self.dt

        # Check whether the friction is greater than the downhill force (e.g., rolling out the ball in the plane)
        if abs(x_sin) < x_friction_coefficient:
            # Check for stopping condition
            if (self.__velocity[0] <= 0 and velocity_x > 0) or (self.__velocity[0] >= 0 and velocity_x < 0):
                x_a = 0
                self.__velocity[0] = 0

        # ---------- y-direction ----------------------------------------------

        y_sin = sin(self.__y_rad)
        y_friction_coefficient = self.__urr * cos(self.__y_rad)

        # Acceleration (sign of friction coefficient depending on direction of speed in y)
        if self.__velocity[1] <= 0:
            y_a = -self.__5_7g * (y_sin - y_friction_coefficient)
        else:
            y_a = -self.__5_7g * (y_sin + y_friction_coefficient)

        # New velocity in y
        velocity_y = self.__velocity[1] + y_a * self.dt

        # Check whether the friction is greater than the downhill force (e.g., rolling out the ball in the plane)
        if abs(y_sin) < y_friction_coefficient:
            # Check for stopping condition
            if (self.__velocity[1] <= 0 and velocity_y > 0) or (self.__velocity[1] >= 0 and velocity_y < 0):
                y_a = 0
                self.__velocity[1] = 0

        # ---------- Update position and velocity -----------------------------

        # Position
        self.__position[0] = self.__position[0] + self.__velocity[0] * self.dt + x_a / 2 * self.dt * self.dt
        self.__position[1] = self.__position[1] + self.__velocity[1] * self.dt + y_a / 2 * self.dt * self.dt

        # Velocity
        self.__velocity[0] = velocity_x
        self.__velocity[1] = velocity_y
        self.__detect_and_process_collision()
        if self.__geometry.layout.number_holes > 0:
            self.__detect_ball_in_hole()

        return self.__position

    # ========== Detect and process collisions and ball in hole ===============
    
    def __detect_and_process_collision(self):
        """
        Detects a collision with a wall and changes the ball behavior if necessary.

        There are two types of collisions to distinguish:
            - collision with an edge
            - collision with a corner

        Parameters
        ----------
        None.

        Returns
        -------
        None.

        """
        damping_factor = 0.1099
        pos_x = self.__position[0]
        pos_y = self.__position[1]
        radius = self.__geometry.ball.radius

        # ---------- Check for collision with an edge -------------------------

        is_collision_edge = False
        for corner in self.__corners:
            # Left edge
            if (pos_x < corner[0].x) and (corner[0].x - pos_x < radius) and (pos_y >= corner[3].y) and (pos_y <= corner[0].y):
                y_friction_coefficient = self.__urr_edge * cos(self.__y_rad)
                y_a = self.__5_7g * y_friction_coefficient                                  # Acceleration in y
                y_dv = (y_a * self.dt) if (self.__velocity[1] <= 0) else (-y_a * self.dt)    # Change of speed in y
                self.__velocity[1] += y_dv
                self.__velocity[0] = -self.__velocity[0] * damping_factor
                self.__position[0] = pos_x = corner[0].x - radius
                is_collision_edge = True
                    
            # Right_edge
            if (pos_x > corner[1].x) and (pos_x - corner[1].x < radius) and (pos_y >= corner[2].y) and (pos_y <= corner[1].y):
                y_friction_coefficient = self.__urr_edge * cos(self.__y_rad)
                y_a = self.__5_7g * y_friction_coefficient                                  # Acceleration in y
                y_dv = (y_a * self.dt) if (self.__velocity[1] <= 0) else (-y_a * self.dt)    # Change of speed in y
                self.__velocity[1] += y_dv
                self.__velocity[0] = -self.__velocity[0] * damping_factor
                self.__position[0] = pos_x = corner[1].x + radius
                is_collision_edge = True
                    
            # Top edge
            if (pos_y > corner[0].y) and (pos_y - corner[0].y < radius) and (pos_x >= corner[0].x) and (pos_x <= corner[1].x):
                x_friction_coefficient = self.__urr_edge * cos(self.__x_rad)
                x_a = self.__5_7g * x_friction_coefficient                                  # Acceleration in x
                x_dv = (x_a * self.dt) if (self.__velocity[0] <= 0) else (-x_a * self.dt)    # Change of speed in x
                self.__velocity[0] += x_dv
                self.__velocity[1] = -self.__velocity[1] * damping_factor
                self.__position[1] = pos_y = corner[0].y + radius
                is_collision_edge = True
                    
            # Bottom edge
            if (pos_y < corner[3].y) and (corner[3].y - pos_y < radius) and (pos_x >= corner[3].x) and (pos_x <= corner[2].x):
                x_friction_coefficient = self.__urr_edge * cos(self.__x_rad)
                x_a = self.__5_7g * x_friction_coefficient                                  # Acceleration in x
                x_dv = (x_a * self.dt) if (self.__velocity[0] <= 0) else (-x_a * self.dt)    # Change of speed in x
                self.__velocity[0] += x_dv
                self.__velocity[1] = -self.__velocity[1] * damping_factor
                self.__position[1] = pos_y = corner[2].y - radius
                is_collision_edge = True

        if (is_collision_edge == True) or (self.__geometry.layout.number_holes == 0): # Check for corner collisions only if there has been no wall collision (preventing misbehavior of the ball) and its the game plate with 0 holes.
            return

        # ---------- Check for collision with a corner ------------------------
        
        is_collision_corner = False
        squared_radius = radius * radius
        for corner in self.__corners:
            # Top left corner (adjust position)
            if ((pos_x - corner[0].x) * (pos_x - corner[0].x) + (pos_y - corner[0].y) * (pos_y - corner[0].y)) < squared_radius:
                sx = pos_x - corner[0].x
                sy = pos_y - corner[0].y
                collision_distance = sqrt(sx * sx + sy * sy)
                distance_factor = radius / collision_distance
                self.__position[0] = pos_x = sx * distance_factor + corner[0].x
                self.__position[1] = pos_y = sy * distance_factor + corner[0].y
                is_collision_corner = True
                break

            # Top right corner (adjust position)
            elif ((pos_x - corner[1].x) * (pos_x - corner[1].x) + (pos_y - corner[1].y) * (pos_y - corner[1].y)) < squared_radius:
                sx = pos_x - corner[1].x
                sy = pos_y - corner[1].y
                collision_distance = sqrt(sx * sx + sy * sy)
                distance_factor = radius/ collision_distance
                self.__position[0] = pos_x = sx * distance_factor + corner[1].x
                self.__position[1] = pos_y = sy * distance_factor + corner[1].y
                is_collision_corner = True
                break

            # Bottom right corner (adjust position)
            elif ((pos_x - corner[2].x) * (pos_x - corner[2].x) + (pos_y - corner[2].y) * (pos_y - corner[2].y)) < squared_radius:
                sx = pos_x - corner[2].x
                sy = pos_y - corner[2].y
                collision_distance = sqrt(sx * sx + sy * sy)
                distance_factor = radius / collision_distance
                self.__position[0] = pos_x = sx * distance_factor + corner[2].x
                self.__position[1] = pos_y = sy * distance_factor + corner[2].y
                is_collision_corner = True
                break

            # Bottom left corner (adjust position)
            elif ((pos_x - corner[3].x) * (pos_x - corner[3].x) + (pos_y - corner[3].y) * (pos_y - corner[3].y)) < squared_radius:
                sx = pos_x - corner[3].x
                sy = pos_y - corner[3].y
                collision_distance = sqrt(sx * sx + sy * sy)
                distance_factor = radius / collision_distance
                self.__position[0] = pos_x = sx * distance_factor + corner[3].x
                self.__position[1] = pos_y = sy * distance_factor + corner[3].y
                is_collision_corner = True
                break

        # Adjust velocity
        if is_collision_corner == True:
            #print("Collision with corner detected")
            collision_unity_vector_x = -sx / collision_distance
            collision_unity_vector_y = -sy / collision_distance
            scalar_product = self.__velocity[0] * collision_unity_vector_x + self.__velocity[1] * collision_unity_vector_y
            collision_velocity_perpendicular_x = scalar_product * collision_unity_vector_x
            collision_velocity_perpendicular_y = scalar_product * collision_unity_vector_y
            self.__velocity[0] -= (1.0 + damping_factor) * collision_velocity_perpendicular_x
            self.__velocity[1] -= (1.0 + damping_factor) * collision_velocity_perpendicular_y

    # -------------------------------------------------------------------------
    
    def __detect_ball_in_hole(self):
        """
        Detects if the ball is falling into a hole.

        Parameters
        ----------
        None.

        Returns
        -------
        None.

        """
        pos_x = self.__position[0]
        pos_y = self.__position[1]
        squared_hole_radius = self.__geometry.holes.radius * self.__geometry.holes.radius

        for hole in self.__geometry.holes.data:
            hole_center = hole["pos"]
            if ((pos_x - hole_center.x) * (pos_x - hole_center.x) + (pos_y - hole_center.y) * (pos_y - hole_center.y)) < squared_hole_radius:
                print("Ball has fallen into a hole")
                self.is_in_hole = True
                break

# -----------------------------------------------------------------------------
# Main (sample)
# -----------------------------------------------------------------------------

if __name__ == '__main__':
    # Init geometry and rendering
    layout = Layout.HOLES_8
    geometry = Geometry(layout=layout)
    ball_start_position = geometry.start_positions[layout]
    render = Render3D(geometry, ball_position=ball_start_position)

    # Init ball physics
    ball_physics = BallPhysics(geometry=geometry, dt=0.01)
    ball_physics.reset(position=ball_start_position)
    time.sleep(2.0)

    # Tilt labyrinth and play steps
    render.rotate_by(x_degree=2.1, y_degree=0)
    x_rad = render.get_x_rad()
    y_rad = render.get_y_rad()
    
    for i in range(150):
        if ball_physics.is_in_hole == False:
            number_steps = 1
            time.sleep(number_steps * ball_physics.dt)
            pos = ball_physics.step(x_rad=x_rad, y_rad=y_rad, number_steps=number_steps)
            render.move_ball(x=pos[0], y=pos[1])
        else:
            render.ball_visibility(False)
            print("Game over")
            break
            
    # Tilt labyrinth and play steps
    render.rotate_by(x_degree= -2.6, y_degree=0.5)
    x_rad = render.get_x_rad()
    y_rad = render.get_y_rad()

    for i in range(500):
        if ball_physics.is_in_hole == False:
            time.sleep(ball_physics.dt)
            pos = ball_physics.step(x_rad=x_rad, y_rad=y_rad)
            render.move_ball(x=pos[0], y=pos[1])
        else:
            render.ball_visibility(False)
            print("Game over")
            break
