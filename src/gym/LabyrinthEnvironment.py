"""
BRIO labyrinth OpenAI gym environment.

The environment follows the gym documentation (last visited: 24.03.2024):
https://gymnasium.farama.org/tutorials/gymnasium_basics/environment_creation/

@authors: Sandra Lassahn, Marc Hensel
@contact: http://www.haw-hamburg.de/marc-hensel
@copyright: 2024
@version: 2024.05.24
@license: CC BY-NC-SA 4.0, see https://creativecommons.org/licenses/by-nc-sa/4.0/deed.en
"""
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from math import pi
import time

from LabyrinthRender3D import LabyrinthRender3D
from LabyrinthGeometry import LabyrinthGeometry
from LabyrinthBallPhysics import LabyrinthBallPhysics

class LabyrinthEnvironment(gym.Env):
    """
    Observation space:
    ------------------
    1. The xy-position of the ball (2 values, np.float32)
    2. The xy-components of the ball's velocity vector (2 values, np.float32)
    3. The field's x- and y-rotation angles in degree (2 values, np.float32)
        
    Action space:
    -------------
    To be defined
    """

    metadata = {'render_modes': ['3D']}

    # ========== Constructor ==================================================

    def __init__(self, layout, render_mode='3D', actions_dt=0.1, physics_dt=0.01):
        """
        Constructor.
        
        The parameters include two time periods dt, where actions_dt specifies
        the frequency with which a reinforcement agent shall take actions,
        while physics_dt specifies the frequency with which ball movements are
        updated. The physics need to be updated in small time intervalls to
        prevent the ball virutalle moving through walls.
        
        For instance:
            actions_dt=0.1 means that an action is taken every 100 ms.
            physics_dt=0.01 means that the ball's position is updated every 10 ms.
        

        Parameters
        ----------
        layout : string
            Layout of holes and walls as defined in LabyrinthGeometry.py. The default is '8 holes'
        render_mode : String, optional
            Render mode to visualize the states (or None). The default is '3D'.
        actions_dt : float, optional
            Time period between steps (i.e., actions taken) [s]. The default is 0.1.
        physics_dt : float, optional
            Time period between simulation steps [s]. The default is 0.01.

        Returns
        -------
        None.

        """
        # Timing
        self.__actions_dt = actions_dt
        self.__physics_dt = physics_dt
        self.__physics_steps_per_action = int(actions_dt / physics_dt)
        self.__last_action_timestamp_sec = 0.0
        
        # Create labyrinth geometry
        geometry = LabyrinthGeometry(layout=layout)
        
        # Field rotation
        self.__x_degree = 0.0
        self.__y_degree = 0.0
                
        # Ball (physics, position, and destination)
        self.__ball_physics = LabyrinthBallPhysics(geometry=geometry, dt=self.__physics_dt)
        self.__ball_start_position = geometry.start_positions[layout]
        self.__ball_position = self.__ball_start_position
        self.__destination_x = geometry.destinations_xy[layout][0]
        self.__destination_y = geometry.destinations_xy[layout][1]

        # Rendering
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode
        if self.render_mode == '3D':
            self.__render_3d = LabyrinthRender3D(geometry)
            self.render()

        # Declare observation space (see class documentation above)
        self.observation_space = spaces.Box(
            low=np.array([-np.inf, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf], dtype=np.float32),
            high=np.array([np.inf, np.inf, np.inf, np.inf, np.inf, np.inf], dtype=np.float32)
        )


        # Declare action space (see class documentation above)
        num_actions_per_component = 9  # There are 9 possible actions per component (x,y)
        self.action_space = spaces.MultiDiscrete([num_actions_per_component, num_actions_per_component]) # MultiDiscrete Aktionsraum erlaubt es, für jede Komponente (x und y) unabhängige diskrete Werte zu definieren. Hier haben beide Komponenten 9 mögliche Werte.
        self.__action_to_angle_degree = [-2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2]

    # ========== Reset ========================================================

    def reset(self, seed=None):
        """
        Reset the environment.

        Parameters
        ----------
        seed : int, optional
            Seed to initialize the random generator. The default is None.

        Returns
        -------
        Dict
            Observation.
        None
            Information (currently not used).

        """
        # Set random seed
        super().reset(seed=seed)

        self.number_actions = 0 #action history counter

        # observation space
        self.observation_space = np.array([
            self.__ball_start_position.x,
            self.__ball_start_position.y,
            0.0,
            0.0,
            0.0,
            0.0
        ], dtype=np.float32)

        # Field rotation
        self.__x_degree = 0.0
        self.__y_degree = 0.0

        # Ball
        self.__ball_position = self.__ball_start_position
        self.__last_ball_position = self.__ball_start_position
        self.__ball_physics.reset(position=self.__ball_start_position)

        # Rendering (ball in hole became invisible)
        if self.render_mode == '3D':
            self.render()
            self.__render_3d.ball_visibility(True)

        #interim reward counter
        self.__interim_reward_counter = 0

        return self.observation_space, {}

    # ========== Rendering ====================================================

    def render(self):
        """
        Render state.

        Returns
        -------
        None.

        """
        if self.render_mode == '3D':
            # Field rotation
            self.__render_3d.rotate_to(self.__x_degree, self.__y_degree)
            
            # Ball visibility
            if self.__ball_physics.is_ball_in_hole:
                self.__render_3d.ball_visibility(False)
                
            # Ball position
            # TODO Check rotation of ball using x_rad and y_rad
            x_rad = self.__x_degree * pi/180.0
            y_rad = self.__y_degree * pi/180.0
            self.__render_3d.move_ball(self.__ball_position.x, self.__ball_position.y, x_rad=x_rad, y_rad=y_rad)
            
            # Update timestamp
            self.__last_render_timestamp_sec = time.time()

    # ========== interim reward ===============================================
    """def interim_reward(self):
        #Für Schwellenbelohnung
        #[Koordinate x oder y, Richtung der Querung, Schwelle, Bereich min, Bereich max]
        # mit Richtung der Querung: 1 = von kleiner zu größeren werten, -1 = von größeren zu kleineren werten
        interim_rewards = [['x', -1, 7.77, -8.16, -3.98],
                           ['x', -1, 3.83, -11.4, -9.59]]
        if len(interim_rewards) <= self.__interim_reward_counter:
            return False

        axis = interim_rewards[self.__interim_reward_counter][0]
        direction = interim_rewards[self.__interim_reward_counter][1]
        threshold = interim_rewards[self.__interim_reward_counter][2]
        range_min = interim_rewards[self.__interim_reward_counter][3]
        range_max = interim_rewards[self.__interim_reward_counter][4]
        if axis == 'x':
            if direction < 0:
                if self.__last_ball_position.x > threshold and self.__ball_position.x <= threshold and self.__ball_position.y > range_min and self.__ball_position.y < range_max:
                    self.__interim_reward_counter += 1
                    return True
            elif self.__last_ball_position.x < threshold and self.__ball_position.x >= threshold and self.__ball_position.y > range_min and self.__ball_position.y < range_max:
                self.__interim_reward_counter += 1
                return True
        else: #axis == 'y'
            if direction < 0:
                if self.__last_ball_position.y > threshold and self.__ball_position.y <= threshold and self.__ball_position.x > range_min and self.__ball_position.x < range_max:
                    self.__interim_reward_counter += 1
                    return True
            elif self.__last_ball_position.y < threshold and self.__ball_position.y >= threshold and self.__ball_position.x > range_min and self.__ball_position.x < range_max:
                self.__interim_reward_counter += 1
                return True
        return False """

    def interim_reward(self):
        # Für Fortschritt

        #Reihnfolge ziel nach anfang
        target_points = [[-4.4, -10.32], [5.82, -10.32], [7.33, -6.88]]
        #x_min, x_max, y_min, y-max
        areas = [[-5.86, 7.46, -11.40, -9.52], [5.21, 7.46, -9.52, -4.63], [7.46, 10.18, -11.4, -4.63]]
        i = 0
        for x_min, x_max, y_min, y_max in areas:
            if x_min < self.__last_ball_position.x and x_max > self.__last_ball_position.x and y_min < self.__last_ball_position.y and y_max > self.__last_ball_position.y:
                last_distance = (self.__last_ball_position.x - target_points[i][0])^2 + (self.__last_ball_position.y - target_points[i][1])^2
                current_distance = (self.__ball_position.x - target_points[i][0]) ^ 2 + (self.__ball_position.y - target_points[i][1]) ^ 2

                if current_distance < last_distance:
                    return True #in die richtige richtung bewegt
                else:
                    return False
            i += 1
        return False



    # ========== Step =========================================================

    def step(self, action):
        """
        Apply an action.

        Parameters
        ----------
        action : TYPE
            DESCRIPTION.

        Returns
        -------
        observation : dict
            Observation after applying the action.
        reward : int
            Reward of applying the action.
        done : boolean
            True if the episode has ended, else False.
        info : dict
            Information (currently not used).

        """
        # Field's rotation angles before and after applying the action
        start_x_degree = self.__x_degree
        start_y_degree = self.__y_degree
        stop_x_degree = self.__action_to_angle_degree[action[0]]
        stop_y_degree = self.__action_to_angle_degree[action[1]]
        is_rotate_field = (stop_x_degree != start_x_degree) or (stop_y_degree != start_y_degree)

        # New ball position
        if is_rotate_field == False:
            # Action does not rotate the field
            x_rad = stop_x_degree * pi/180.0
            y_rad = stop_y_degree * pi/180.0
            self.__ball_position = self.__ball_physics.step(x_rad=x_rad, y_rad=y_rad, number_steps=self.__physics_steps_per_action)
        else:            
            # Actions does rotates the field linearly
            #Wenn das Spielfeld rotiert, wird die Ballposition in mehreren Schritten von der Startposition zur Endposition linear interpoliert und jeweils aktualisiert.
            start_x_rad = start_x_degree * pi/180.0
            start_y_rad = start_y_degree * pi/180.0
            stop_x_rad = stop_x_degree * pi/180.0
            stop_y_rad = stop_y_degree * pi/180.0
            delta_x_rad = (stop_x_rad - start_x_rad) / self.__physics_steps_per_action
            delta_y_rad = (stop_y_rad - start_y_rad) / self.__physics_steps_per_action
            
            for i in range (self.__physics_steps_per_action):
                x_rad = start_x_rad + i * delta_x_rad
                y_rad = start_y_rad + i * delta_y_rad
                self.__ball_position = self.__ball_physics.step(x_rad=x_rad, y_rad=y_rad)

        # Store field's final rotation
        self.__x_degree = stop_x_degree
        self.__y_degree = stop_y_degree
        
        # Rendering active (Wait until period between steps has passed and render)
        if self.render_mode == '3D':
            timestamp = time.time()
            elapsed_time = timestamp - self.__last_action_timestamp_sec
            wait_time = float(max(self.__actions_dt - elapsed_time, 0))
            time.sleep(wait_time)
            self.__last_action_timestamp_sec = timestamp + wait_time
            self.render()
            
        # Observation_space
        self.observation_space = np.array([
            self.__ball_start_position.x,
            self.__ball_start_position.y,
            self.__ball_physics.get_velocity().x,
            self.__ball_physics.get_velocity().y,
            self.__x_degree,
            self.__y_degree
        ], dtype=np.float32)

        # Ball reached destination?
        is_destination_x = (self.__ball_position.x > self.__destination_x[0]) and (self.__ball_position.x < self.__destination_x[1])
        is_destination_y = (self.__ball_position.y > self.__destination_y[0]) and (self.__ball_position.y < self.__destination_y[1])
        is_ball_at_destination = is_destination_x and is_destination_y

        # Ball has fallen into a hole?
        is_ball_in_hole = self.__ball_physics.is_ball_in_hole

        # Reward
        if is_ball_at_destination:
            print("Ball reached destination")
            reward = 500
        elif is_ball_in_hole:
            print("Ball lost")
            reward = -1000
        elif self.interim_reward():
            print("interim reward")
            reward = 1
        else:
            reward = -1

        # Episode completed or truncated?
        done = is_ball_at_destination or is_ball_in_hole
        self.number_actions += 1 # Action history
        if self.number_actions >= 10000:
            truncated = True
        else:
            truncated = False

        return self.observation_space, reward, done, truncated, {}


# -----------------------------------------------------------------------------
# Main (sample)
# -----------------------------------------------------------------------------

if __name__ == '__main__':
    env = LabyrinthEnvironment(layout='8 holes', render_mode='3D')
    env.reset()

    for action in [[2,3], [6,3], [8,4], [7,5]]:
        env.step(action)
    for action in range(4):
        env.step([7,4])
    for action in range(20):
        env.step([3,8])
