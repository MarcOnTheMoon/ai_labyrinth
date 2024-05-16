"""
Models realistic behavior of the ball on the field for a labyrinth OpenAI gym environment.

The graphical user interface is based on VPython ( https://vpython.org/ ).
The library uses the default browser to display images and animations. Install
VPython in Anaconda by the command 'conda install -c conda-forge vpython'.

@authors: Sandra Lassahn, Marc Hensel
@contact: http://www.haw-hamburg.de/marc-hensel
@copyright: 2024
@version: 2024.05.16
@license: CC BY-NC-SA 4.0, see https://creativecommons.org/licenses/by-nc-sa/4.0/deed.en
"""
from vpython import vector as vec
from math import sqrt, sin, cos
import time
from LabyrinthRender3D import LabyrinthRender3D
from LabyrinthGeometry import LabyrinthGeometry

class LabyrinthBallPhysics:

    # ========== Constructor ==================================================

    def __init__(self, geometry, time_step_secs):
        """
        Constructor.

        Parameters
        ----------
        geometry : LabyrinthGeometry
            Geometry for calculations
        time_step_secs : float
            Time period between steps [s]

        Returns
        -------
        None.

        """
        # Timing
        self.dt = time_step_secs                # Time step [s]

        # Ball state
        # TODO Component z used?
        self.is_ball_in_hole = False            # Ball has fallen into hole if True
        self.__position = vec(0.0, 0.0, 0.0)    # Ball's position [cm]^3
        self.__velocity = vec(0.0, 0.0, 0.0)    # Ball's velocity [cm/s]^3
        
        # Physical constants
        self.__g = 9.81 * 100.0                 # Acceleration of gravity [cm/s²]
        self.__5_7g = 5.0 / 7.0 * self.__g      # constant for ball acceleration [cm/s²]
        self.__urr = 0.00118                    # Rolling friction coefficient
        self.__urr_edge = 0.00212               # Friction coefficients for collision with an edge

        # Geometry        
        self.__geometry = geometry
        self.__init_corners()

    # ========== Public methods ===============================================
        
    def get_velocity(self):
        """
        Get the ball's velocity vector.

        Returns
        -------
        vpython.vector
            3D velocity vector.

        """
        return self.__velocity

    # -------------------------------------------------------------------------
    
    def reset(self, position):
        """
        Reset ball physics.

        Parameters
        ----------
        position : vpython.vector
            Position of ball.

        Returns
        -------
        None.

        """
        self.__position = position
        self.__velocity = vec(0.0, 0.0, 0.0)
        self.is_ball_in_hole = False

    # -------------------------------------------------------------------------
    
    def move_one_time_step(self, x_rad, y_rad, position=None):
        """
        Calculates the new velocity and position of the ball for the next time step.
        And call the functions self.__detect_and_process_collisions() and self.__detect_ball_in_hole()

        Parameters
        ----------
        x_rad : float
            Field angle in x-direction [rad]
        y_rad : float
            Field angle in y-direction [rad]
        position : vec(x,y,z), optional
            Start position of the Ball. The default is None.

        Returns
        -------
        position : vec(x,y,z)
            new ball position

        """
        # Set arguments
        self.__x_rad = x_rad
        self.__y_rad = y_rad

        if position != None:
            self.__position = position

        # ---------- x-direction ----------------------------------------------

        x_sin = sin(self.__x_rad)
        x_friction_coefficient = self.__urr * cos(self.__x_rad)

        # Acceleration (sign of friction coefficient depending on direction of speed in x)
        if self.__velocity.x <= 0:
            x_a = -self.__5_7g * (x_sin - x_friction_coefficient)
        else:
            x_a = -self.__5_7g * (x_sin + x_friction_coefficient)

        # New velocity in x
        velocity_x = self.__velocity.x + x_a * self.dt

        # Check whether the friction is greater than the downhill force (e.g., rolling out the ball in the plane)
        if abs(x_sin) < x_friction_coefficient:
            # Check for stopping condition
            if (self.__velocity.x <= 0 and velocity_x > 0) or (self.__velocity.x >= 0 and velocity_x < 0):
                x_a = 0
                self.__velocity.x = 0

        # ---------- y-direction ----------------------------------------------

        y_sin = sin(self.__y_rad)
        y_friction_coefficient = self.__urr * cos(self.__y_rad)

        # Acceleration (sign of friction coefficient depending on direction of speed in y)
        if self.__velocity.y <= 0:
            y_a = -self.__5_7g * (y_sin - y_friction_coefficient)
        else:
            y_a = -self.__5_7g * (y_sin + y_friction_coefficient)

        # New velocity in y
        velocity_y = self.__velocity.y + y_a * self.dt

        # Check whether the friction is greater than the downhill force (e.g., rolling out the ball in the plane)
        if abs(y_sin) < y_friction_coefficient:
            # Check for stopping condition
            if (self.__velocity.y <= 0 and velocity_y > 0) or (self.__velocity.y >= 0 and velocity_y < 0):
                y_a = 0
                self.__velocity.y = 0

        # ---------- Update position and velocity -----------------------------

        # Position
        position_x = self.__position.x + self.__velocity.x * self.dt + x_a / 2 * self.dt * self.dt
        position_y = self.__position.y + self.__velocity.y * self.dt + y_a / 2 * self.dt * self.dt
        self.__position = vec(position_x, position_y, 0.0)

        # Velocity
        self.__velocity = vec(velocity_x, velocity_y, 0.0)
        self.__detect_and_process_collisions()
        self.__detect_ball_in_hole()

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
        # Interior walls (copied from LabyrinthGeometry)
        walls_data = self.__geometry.walls.data.copy()
        
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

    # ========== Detect and process collisions and ball in hole ===============
    
    def __detect_and_process_collisions(self):
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
        pos_x = self.__position.x
        pos_y = self.__position.y
        radius = self.__geometry.ball.radius

        # ---------- Check for collision with an edge -------------------------

        is_collision_edge = False
        for corner in self.__corners:
            # Left edge
            if (pos_x < corner[0].x) and (corner[0].x - pos_x < radius) and (pos_y >= corner[3].y) and (pos_y <= corner[0].y):
                y_friction_coefficient = self.__urr_edge * cos(self.__y_rad)
                y_a = self.__5_7g * y_friction_coefficient                                  # Acceleration in y
                y_dv = (y_a * self.dt) if (self.__velocity.y <= 0) else (-y_a * self.dt)    # Change of speed in y
                self.__velocity.y += y_dv
                self.__velocity.x = -self.__velocity.x * damping_factor
                self.__position.x = pos_x = corner[0].x - radius
                is_collision_edge = True
                break
                    
            # Right_edge
            if (pos_x > corner[1].x) and (pos_x - corner[1].x < radius) and (pos_y >= corner[2].y) and (pos_y <= corner[1].y):
                y_friction_coefficient = self.__urr_edge * cos(self.__y_rad)
                y_a = self.__5_7g * y_friction_coefficient                                  # Acceleration in y
                y_dv = (y_a * self.dt) if (self.__velocity.y <= 0) else (-y_a * self.dt)    # Change of speed in y
                self.__velocity.y += y_dv
                self.__velocity.x = -self.__velocity.x * damping_factor
                self.__position.x = pos_x = corner[1].x + radius
                is_collision_edge = True
                break
                    
            # Top edge
            if (pos_y > corner[0].y) and (pos_y - corner[0].y < radius) and (pos_x >= corner[0].x) and (pos_x <= corner[1].x):
                x_friction_coefficient = self.__urr_edge * cos(self.__x_rad)
                x_a = self.__5_7g * x_friction_coefficient                                  # Acceleration in x
                x_dv = (x_a * self.dt) if (self.__velocity.x <= 0) else (-x_a * self.dt)    # Change of speed in x
                self.__velocity.x += x_dv
                self.__velocity.y = -self.__velocity.y * damping_factor
                self.__position.y = pos_y = corner[0].y + radius
                is_collision_edge = True
                break
                    
            # Bottom edge
            if (pos_y < corner[3].y) and (corner[3].y - pos_y < radius) and (pos_x >= corner[3].x) and (pos_x <= corner[2].x):
                x_friction_coefficient = self.__urr_edge * cos(self.__x_rad)
                x_a = self.__5_7g * x_friction_coefficient                                  # Acceleration in x
                x_dv = (x_a * self.dt) if (self.__velocity.x <= 0) else (-x_a * self.dt)    # Change of speed in x
                self.__velocity.x += x_dv
                self.__velocity.y = -self.__velocity.y * damping_factor
                self.__position.y = pos_y = corner[2].y - radius
                is_collision_edge = True
                break


        if is_collision_edge == True: # Check for corner collisions only if there has been no wall collision (preventing misbehavior of the ball).
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
                self.__position.x = pos_x = sx * distance_factor + corner[0].x
                self.__position.y = pos_y = sy * distance_factor + corner[0].y
                is_collision_corner = True
                break

            # Top right corner (adjust position)
            elif ((pos_x - corner[1].x) * (pos_x - corner[1].x) + (pos_y - corner[1].y) * (pos_y - corner[1].y)) < squared_radius:
                sx = pos_x - corner[1].x
                sy = pos_y - corner[1].y
                collision_distance = sqrt(sx * sx + sy * sy)
                distance_factor = radius/ collision_distance
                self.__position.x = pos_x = sx * distance_factor + corner[1].x
                self.__position.y = pos_y = sy * distance_factor + corner[1].y
                is_collision_corner = True
                break

            # Bottom right corner (adjust position)
            elif ((pos_x - corner[2].x) * (pos_x - corner[2].x) + (pos_y - corner[2].y) * (pos_y - corner[2].y)) < squared_radius:
                sx = pos_x - corner[2].x
                sy = pos_y - corner[2].y
                collision_distance = sqrt(sx * sx + sy * sy)
                distance_factor = radius / collision_distance
                self.__position.x = pos_x = sx * distance_factor + corner[2].x
                self.__position.y = pos_y = sy * distance_factor + corner[2].y
                is_collision_corner = True
                break

            # Bottom left corner (adjust position)
            elif ((pos_x - corner[3].x) * (pos_x - corner[3].x) + (pos_y - corner[3].y) * (pos_y - corner[3].y)) < squared_radius:
                sx = pos_x - corner[3].x
                sy = pos_y - corner[3].y
                collision_distance = sqrt(sx * sx + sy * sy)
                distance_factor = radius / collision_distance
                self.__position.x = pos_x = sx * distance_factor + corner[3].x
                self.__position.y = pos_y = sy * distance_factor + corner[3].y
                is_collision_corner = True
                break

        # Adjust velocity
        if is_collision_corner == True:
            print("Collision with corner detected")
            collision_unity_vector_x = -sx / collision_distance
            collision_unity_vector_y = -sy / collision_distance
            scalar_product = self.__velocity.x * collision_unity_vector_x + self.__velocity.y * collision_unity_vector_y
            collision_velocity_perpendicular_x = scalar_product * collision_unity_vector_x
            collision_velocity_perpendicular_y = scalar_product * collision_unity_vector_y
            self.__velocity.x -= (1.0 + damping_factor) * collision_velocity_perpendicular_x
            self.__velocity.y -= (1.0 + damping_factor) * collision_velocity_perpendicular_y

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
        pos_x = self.__position.x
        pos_y = self.__position.y
        squared_hole_radius = self.__geometry.holes.radius * self.__geometry.holes.radius

        for hole in self.__geometry.holes.data:
            hole_center = hole["pos"]
            if ((pos_x - hole_center.x) * (pos_x - hole_center.x) + (pos_y - hole_center.y) * (pos_y - hole_center.y)) < squared_hole_radius:
                print("Ball has fallen into a hole")
                self.is_ball_in_hole = True
                break

# -----------------------------------------------------------------------------
# Main (sample)
# -----------------------------------------------------------------------------

if __name__ == '__main__':
    # Init geometry and rendering
    layout = '8 holes'
    geometry = LabyrinthGeometry(layout=layout)
    ball_start_position = geometry.start_positions[layout]
    render = LabyrinthRender3D(geometry, ball_position=ball_start_position)

    # Init ball physics (including position and tilting)
    x_rad = render.get_x_rad()
    y_rad = render.get_y_rad()
    ball_physics = LabyrinthBallPhysics(geometry=geometry, time_step_secs=0.01)
    ball_physics.move_one_time_step(x_rad=x_rad, y_rad=y_rad, position=ball_start_position)
    time.sleep(2.0)

    # Tilt labyrinth and play steps
    render.rotate_by(x_degree=2.0, y_degree=0.5)
    x_rad = render.get_x_rad()
    y_rad = render.get_y_rad()
    
    for i in range(50):
        if ball_physics.is_ball_in_hole == False:
            time.sleep(ball_physics.dt)
            pos = ball_physics.move_one_time_step(x_rad=x_rad, y_rad=y_rad)
            render.move_ball(x=pos.x, y=pos.y, x_rad=x_rad, y_rad=y_rad)
        else:
            render.ball_visibility(False)
            print("Game over")
            break
            
    # Tilt labyrinth and play steps
    render.rotate_by(x_degree= -2.5, y_degree=0.5)
    x_rad = render.get_x_rad()
    y_rad = render.get_y_rad()

    for i in range(2000):
        if ball_physics.is_ball_in_hole == False:
            time.sleep(ball_physics.dt)
            pos = ball_physics.move_one_time_step(x_rad=x_rad, y_rad=y_rad)
            render.move_ball(x=pos.x, y=pos.y, x_rad=x_rad, y_rad=y_rad)
        else:
            render.ball_visibility(False)
            print("Game over")
            break
