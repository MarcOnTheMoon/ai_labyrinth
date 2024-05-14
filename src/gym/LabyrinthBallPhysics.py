"""
Models realistic behavior of the ball on the field for a labyrinth OpenAI gym environment.

The graphical user interface is based on VPython ( https://vpython.org/ ).
The library uses the default browser to display images and animations. Install
VPython in Anaconda by the command 'conda install -c conda-forge vpython'.

@authors: Sandra Lassahn, Marc Hensel
@contact: http://www.haw-hamburg.de/marc-hensel
@copyright: 2024
@version: 2024.05.14
@license: CC BY-NC-SA 4.0, see https://creativecommons.org/licenses/by-nc-sa/4.0/deed.en
"""
from vpython import vector as vec
from math import sqrt, sin, cos
import time
from LabyrinthRender3D import LabyrinthRender3D
from LabyrinthGeometry import LabyrinthGeometry

class LabyrinthBallPhysics:

    def __init__(self, geometry, time_step_secs):
        """
        Constructor.

        Parameters
        ----------
        geometry : LabyrinthGeometry
            Geometry for calculations

        Returns
        -------
        None.

        """
        # TODO Physical unit of velocity?
        self.__geometry = geometry
        self.game_state = 0                     # 0 init state, -1 ball lost
        self.dt = time_step_secs                # Time step [s]
        self.__g = 9.81 * 100.0                 # Acceleration of gravity [cm/s²]
        self.__5_7g = 5.0 / 7.0 * self.__g      # constant for ball acceleration
        self.__urr = 0.00118                    # Rolling friction coefficient
        self.__velocity = vec(0.0, 0.0, 0.0)    # Ball's velocity
        self.__position = vec(0.0, 0.0, 0.0)    # Ball's position
        self.calc_corners()

    # -------------------------------------------------------------------------
    
    def calc_corners(self):
        """
        Calculates all corner points of the individual labyrinth walls.

        Parameters
        ----------
        None.

        Returns
        -------
        None.

        """
        #interior walls
        walls_data = self.__geometry.walls.data
        #exterior walls
        field_x = self.__geometry.field.size_x
        field_y = self.__geometry.field.size_y
        height = self.__geometry.box.height
        thickness = self.__geometry.box.boarder
        walls_data.append({"pos": vec(0,  (field_y + thickness)/2, 0), "size": vec(field_x + 2 * thickness, thickness, height)}) #upper
        walls_data.append({"pos": vec(0, -(field_y + thickness)/2, 0), "size": vec(field_x + 2 * thickness, thickness, height)}) #lower
        walls_data.append({"pos": vec(-(field_x + thickness)/2, 0, 0), "size": vec(thickness, field_y, height)}) #left
        walls_data.append({"pos": vec((field_x + thickness) / 2, 0, 0), "size": vec(thickness, field_y, height)}) #right

        #corner calculation
        self.__corners = []
        for wall_data in walls_data:
            wall_center = wall_data["pos"]
            wall_size = wall_data["size"]
            corner_left_upper = vec( wall_center.x - wall_size.x /2, wall_center.y + wall_size.y /2, wall_center.z)
            corner_right_upper = vec(wall_center.x + wall_size.x / 2, wall_center.y + wall_size.y / 2, wall_center.z)
            corner_right_lower = vec(wall_center.x + wall_size.x / 2, wall_center.y - wall_size.y / 2, wall_center.z)
            corner_left_lower = vec(wall_center.x - wall_size.x / 2, wall_center.y - wall_size.y / 2, wall_center.z)

            self.__corners.append([corner_left_upper, corner_right_upper, corner_right_lower, corner_left_lower])

    # -------------------------------------------------------------------------
    def collision_wall(self):
        """
        Detects a collision with a wall and changes the ball behavior if necessary.

        There are two types of collisions to distinguish:
            - collision with an edge and
            - collision with a corner.

        Parameters
        ----------
        None.

        Returns
        -------
        None.

        """
        damping_factor = 0.1099
        self.__urr_edge = 0.00212 # difference of friction coefficients
        collision_edge = 0

        # ===== collision with an edge =====

        for corner_data in self.__corners:
            if self.__position.x < corner_data[0].x:
                #left_edge
                if (corner_data[0].x - self.__position.x < self.__geometry.ball.radius) and ((self.__position.y >= corner_data[3].y) and (self.__position.y <= corner_data[0].y)):
                    collision_edge = 1
                    y_friction_coefficient = self.__urr_edge * cos(self.__y_rad)
                    if self.__velocity.y <= 0:
                        y_a = self.__5_7g * y_friction_coefficient  # acceleration in x
                    else:
                        y_a = -self.__5_7g * y_friction_coefficient  # acceleration in x
                    self.__velocity.y = self.__velocity.y + y_a * self.dt
                    self.__velocity.x = -self.__velocity.x * damping_factor

                    self.__position.x = corner_data[0].x - self.__geometry.ball.radius
            if self.__position.x > corner_data[1].x:
                # right_edge
                if (self.__position.x - corner_data[1].x < self.__geometry.ball.radius) and ((self.__position.y >= corner_data[2].y) and (self.__position.y <= corner_data[1].y)):
                    collision_edge = 2
                    y_friction_coefficient = self.__urr_edge * cos(self.__y_rad)
                    if self.__velocity.y <= 0:
                        y_a = self.__5_7g * y_friction_coefficient  # acceleration in x
                    else:
                        y_a = -self.__5_7g * y_friction_coefficient  # acceleration in x
                    self.__velocity.y = self.__velocity.y + y_a * self.dt
                    self.__velocity.x = -self.__velocity.x * damping_factor
                    self.__position.x = corner_data[1].x + self.__geometry.ball.radius
            if self.__position.y > corner_data[0].y:
                #edge_top
                if (self.__position.y - corner_data[0].y < self.__geometry.ball.radius) and ((self.__position.x >= corner_data[0].x) and (self.__position.x <= corner_data[1].x)):
                    collision_edge = 3
                    x_friction_coefficient = self.__urr_edge * cos(self.__x_rad)
                    if self.__velocity.x <= 0:
                        x_a = self.__5_7g * x_friction_coefficient  # acceleration in x
                    else:
                        x_a = -self.__5_7g * x_friction_coefficient  # acceleration in x
                    self.__velocity.x = self.__velocity.x + x_a * self.dt
                    self.__velocity.y = -self.__velocity.y * damping_factor

                    self.__position.y = corner_data[0].y + self.__geometry.ball.radius
            if self.__position.y < corner_data[3].y:
                #edge_buttom
                if (corner_data[3].y - self.__position.y < self.__geometry.ball.radius) and ((self.__position.x >= corner_data[3].x) and (self.__position.x <= corner_data[2].x)):
                    collision_edge = 4
                    x_friction_coefficient = self.__urr_edge * cos(self.__x_rad)
                    if self.__velocity.x <= 0:
                        x_a = self.__5_7g * x_friction_coefficient  # acceleration in x
                    else:
                        x_a = -self.__5_7g * x_friction_coefficient  # acceleration in x
                    self.__velocity.x = self.__velocity.x + x_a * self.dt
                    self.__velocity.y = -self.__velocity.y * damping_factor

                    self.__position.y = corner_data[2].y - self.__geometry.ball.radius



        # ===== collision with a corner =====

        collision_corner = 0
        if collision_edge == 0: #Check for corner collisions only if there has been no wall collision (preventing misbehavior of the ball).
            for corner_data in self.__corners:
                #corner left top
                if ((self.__position.x-corner_data[0].x) * (self.__position.x-corner_data[0].x) + (self.__position.y-corner_data[0].y) * (self.__position.y-corner_data[0].y)) < self.__geometry.ball.radius * self.__geometry.ball.radius:
                    collision_corner = 1
                    # position adjustment
                    sx = self.__position.x-corner_data[0].x
                    sy = self.__position.y-corner_data[0].y
                    collision_distance = sqrt((sx) * (sx) + (sy) * (sy))
                    distance_factor = self.__geometry.ball.radius / collision_distance
                    self.__position.x = sx * distance_factor + corner_data[0].x
                    self.__position.y = sy * distance_factor + corner_data[0].y

                #corner right top
                elif ((self.__position.x-corner_data[1].x) * (self.__position.x-corner_data[1].x) + (self.__position.y-corner_data[1].y) * (self.__position.y-corner_data[1].y)) < self.__geometry.ball.radius * self.__geometry.ball.radius:
                    collision_corner = 2
                    #position adjustment
                    sx = self.__position.x-corner_data[1].x
                    sy = self.__position.y-corner_data[1].y
                    collision_distance = sqrt((sx) * (sx) + (sy) * (sy))
                    distance_factor = self.__geometry.ball.radius/ collision_distance
                    self.__position.x = sx * distance_factor + corner_data[1].x
                    self.__position.y = sy * distance_factor + corner_data[1].y

                #corner right buttom
                elif ((self.__position.x-corner_data[2].x) * (self.__position.x-corner_data[2].x) + (self.__position.y-corner_data[2].y) * (self.__position.y-corner_data[2].y)) < self.__geometry.ball.radius * self.__geometry.ball.radius:
                    collision_corner = 3
                    # position adjustment
                    sx = self.__position.x-corner_data[2].x
                    sy = self.__position.y-corner_data[2].y
                    collision_distance = sqrt((sx) * (sx) + (sy) * (sy))
                    distance_factor = self.__geometry.ball.radius / collision_distance
                    self.__position.x = sx * distance_factor + corner_data[2].x
                    self.__position.y = sy * distance_factor + corner_data[2].y

                #corner left buttom
                elif ((self.__position.x-corner_data[3].x) * (self.__position.x-corner_data[3].x) + (self.__position.y-corner_data[3].y) * (self.__position.y-corner_data[3].y)) < self.__geometry.ball.radius * self.__geometry.ball.radius:
                    collision_corner = 4
                    # position adjustment
                    sx = self.__position.x-corner_data[3].x
                    sy = self.__position.y-corner_data[3].y
                    collision_distance = sqrt((sx) * (sx) + (sy) * (sy))
                    distance_factor = self.__geometry.ball.radius / collision_distance
                    self.__position.x = sx * distance_factor + corner_data[3].x
                    self.__position.y = sy * distance_factor + corner_data[3].y


            if collision_corner != 0:
                print("cornercollision")
                # velocity adjustmant
                collision_unity_vector_x = -sx / collision_distance
                collision_unity_vector_y = -sy / collision_distance
                scalar_product = self.__velocity.x * collision_unity_vector_x + self.__velocity.y * collision_unity_vector_y
                collision_velocity_perpendicular_x = scalar_product * collision_unity_vector_x
                collision_velocity_perpendicular_y = scalar_product * collision_unity_vector_y
                self.__velocity.x -= (1.0 + damping_factor) * collision_velocity_perpendicular_x
                self.__velocity.y -= (1.0 + damping_factor) * collision_velocity_perpendicular_y

    # -------------------------------------------------------------------------
    def collision_hole(self):
        """
        Detects a collision with a hole.

        Parameters
        ----------
        None.

        Returns
        -------
        None.

        """

        holes_data = self.__geometry.holes.data
        for hole_data in holes_data:
            hole_center = hole_data["pos"]
            if ((self.__position.x - hole_center.x) * (self.__position.x - hole_center.x) + (self.__position.y - hole_center.y) * (self.__position.y - hole_center.y)) < self.__geometry.holes.radius * self.__geometry.holes.radius:
                print("holecollision")
                self.game_state = -1


    # -------------------------------------------------------------------------
    def calc_move(self, x_rad, y_rad, position=None):
        """
        Calculates the new velocity and position of the ball for the next time step.
        And call the functions self.collision_wall() and self.collision_hole()

        Parameters
        ----------
        x_rad : float
            field angle in x-direction [rad]
        y_rad : float
            field angle in y-direction [rad]
        position : vec(x,y,z), optional
            start position of the Ball. The default is None.

        Returns
        -------
        position : vec(x,y,z)
            new ball position

        """
        self.__x_rad = x_rad
        self.__y_rad = y_rad

        if position != None:
            self.__position = position

        # ===== x-direction =====

        x_sin = sin(self.__x_rad)
        x_friction_coefficient = self.__urr * cos(self.__x_rad)

        #Consider the sign of the coefficient of friction, depending on the direction of speed in x
        if self.__velocity.x <= 0:
            x_a = -self.__5_7g * (x_sin - x_friction_coefficient)  # acceleration in x
        else:
            x_a = -self.__5_7g * (x_sin + x_friction_coefficient)  # acceleration in x

        #new velocity in x
        velocity_x = self.__velocity.x + x_a * self.dt

        #Check whether the friction is greater than the downhill force. Example application: rolling out the ball in the plane
        if abs(x_sin) < x_friction_coefficient:
            #Check for stopping condition
            if (self.__velocity.x <= 0 and velocity_x > 0) or (self.__velocity.x >= 0 and velocity_x < 0):
                x_a = 0
                self.__velocity.x = 0

        # ===== y-direction =====

        y_sin = sin(self.__y_rad)
        y_friction_coefficient = self.__urr * cos(self.__y_rad)

        # Consider the sign of the coefficient of friction, depending on the direction of speed in y
        if self.__velocity.y <= 0:
            y_a = -self.__5_7g * (y_sin - y_friction_coefficient)  # acceleration in y
        else:
            y_a = -self.__5_7g * (y_sin + y_friction_coefficient)  # acceleration in y

        # new velocity in y
        velocity_y = self.__velocity.y + y_a * self.dt

        # Check whether the friction is greater than the downhill force. Example application: rolling out the ball in the plane
        if abs(y_sin) < y_friction_coefficient:
            # Check for stopping condition
            if (self.__velocity.y <= 0 and velocity_y > 0) or (self.__velocity.y >= 0 and velocity_y < 0):
                y_a = 0
                self.__velocity.y = 0


        # new position
        position_x = self.__position.x + self.__velocity.x * self.dt + x_a / 2 * self.dt * self.dt
        position_y = self.__position.y + self.__velocity.y * self.dt + y_a / 2 * self.dt * self.dt
        self.__position = vec(position_x, position_y, 0.0)

        # update new velocity
        self.__velocity = vec(velocity_x, velocity_y, 0.0)
        self.collision_wall()
        self.collision_hole()
        return self.__position

    def get_velocity(self):
        return self.__velocity

# -----------------------------------------------------------------------------
# Main (sample)
# -----------------------------------------------------------------------------

if __name__ == '__main__':
    # Test
    # Render initial labyrinth environment
    geometry = LabyrinthGeometry()
    #ball_start_position = vec(-1.52, 9.25, 0) #field with 2 holes
    ball_start_position = vec(0.13, 10.53, 0)  # field with 8 holes
    render = LabyrinthRender3D(geometry, ball_position=ball_start_position)

    time.sleep(2.0)
    myBall = LabyrinthBallPhysics(geometry=geometry, time_step_secs=0.01)

    render.rotate_by(x_degree=0, y_degree=0)
    render.ball_visibility(True)


    x_rad = render.get_x_rad()
    y_rad = render.get_y_rad()
    pos = myBall.calc_move(x_rad, y_rad, ball_start_position)

    render.rotate_by(x_degree=2, y_degree=0.5)
    x_rad = render.get_x_rad()
    y_rad = render.get_y_rad()
    for i in range(50):
        if myBall.game_state != -1:
            time.sleep(myBall.dt)
            pos = myBall.calc_move(x_rad, y_rad)
            print("pos",pos)
            render.move_ball(pos.x, pos.y, x_rad=x_rad, y_rad=y_rad)
    if myBall.game_state != -1:
        render.rotate_by(x_degree= -2.5, y_degree=0.5)
        x_rad = render.get_x_rad()
        y_rad = render.get_y_rad()

    for i in range(2000):
        if myBall.game_state != -1:
            time.sleep(myBall.dt)
            pos = myBall.calc_move(x_rad, y_rad)
            print("pos",pos)
            render.move_ball(pos.x, pos.y, x_rad=x_rad, y_rad=y_rad)


    if myBall.game_state == -1:
        render.rotate_to(0,0)
        print("gameover")
        render.ball_visibility(False)