"""
Labyrinth OpenAI gym environment.

The environment follows the gym documentation (last visited: 24.03.2024):
https://gymnasium.farama.org/tutorials/gymnasium_basics/environment_creation/

@authors: Sandra Lassahn, Marc Hensel
@contact: http://www.haw-hamburg.de/marc-hensel
@copyright: 2024
@version: 2024.08.26
@license: CC BY-NC-SA 4.0, see https://creativecommons.org/licenses/by-nc-sa/4.0/deed.en
"""
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random
import time
import math
from math import pi

from LabRender3D import Render3D
from LabLayouts import Layout, Geometry
from LabBallPhysics import BallPhysics
from LabRewards import RewardsByAreas

# TODO Numpy array significantly faster and less memory than Python lists => Replace with Numpy where appropriate

class LabyrinthEnv(gym.Env):
    """
    Observation space:
    ------------------
    1. The xy-position of the ball (2 values, np.float32)
    2. The last xy-position of the ball (2 values, np.float32)
    3. The field's x- and y-rotation angles in degree (2 values, np.float32)
        
    Action space:
    -------------
    A Discrete space where the number of actions is `2 * self.num_actions_per_component`.
    `self.num_actions_per_component` is the number of discrete angle options per x- and y-component,
    there are 5 to 7 components per axis.
    """

    metadata = {'render_modes': ['3D']}

    # ========== Constructor ==================================================

    def __init__(self, layout, render_mode='3D', actions_dt=0.1, physics_dt=0.01):
        """
        Constructor.
        
        The parameters include two time periods dt, where actions_dt specifies
        the frequency with which a reinforcement agent shall take actions,
        while physics_dt specifies the frequency with which ball movements are
        updated. The physics need to be updated in smaller time intervals to
        prevent the ball virtual moving through walls.
        
        For instance:
            actions_dt=0.1 means that an action is taken every 100 ms.
            physics_dt=0.01 means that the ball's position is updated every 10 ms.

        Parameters
        ----------
        layout : enum Layout
            Layout of holes and walls as defined in LabLayouts.py.
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
        assert type(layout) == Layout
        self.__geometry = Geometry(layout=layout)

        # Object to determine rewards of an step
        self.__rewards_rules = RewardsByAreas(layout=layout)

        # Field rotation
        self.__x_degree = 0.0
        self.__y_degree = 0.0
                
        # Ball (physics, position, and destination)
        self.__ball_physics = BallPhysics(geometry=self.__geometry, dt=self.__physics_dt)
        self.__ball_start_position = self.__geometry.start_positions[layout]
        self.__ball_position = self.__ball_start_position
        self.__destination_x = self.__geometry.destinations_xy[layout][0]
        self.__destination_y = self.__geometry.destinations_xy[layout][1]

        # Rendering
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.__render_mode = render_mode
        if self.__render_mode == '3D':
            self.__render_3d = Render3D(self.__geometry)
            self.__render()

        # Declare observation space (see class documentation above)
        self.__observation_space = spaces.Box(
            low=np.array([-np.inf, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf], dtype=np.float32),
            high=np.array([np.inf, np.inf, np.inf, np.inf, np.inf, np.inf], dtype=np.float32)
        )

        # Declare action space (see class documentation above)
        # TODO Why different action space for Layout.HOLES_2? This is not consistent.
        self.__action_to_angle_degree = np.array([-1, -0.5, 0, 0.5, 1], dtype=np.float32)
        if (self.__geometry.layout == Layout.HOLES_2):
            self.__action_to_angle_degree = np.array([-1.5, -1, -0.5, 0, 0.5, 1, 1.5], dtype=np.float32)
        self.num_actions_per_component = len(self.__action_to_angle_degree)

        # Define max. actions per episode for truncated
        # TODO Why distinguish? Would there be any harm in setting highest value 800 for all cases?
        # TODO What about Layout.HOLES_21?
        if self.__geometry.layout.number_holes == 0:
            self.__max_number_actions = 300
        elif self.__geometry.layout.number_holes == 2:
            self.__max_number_actions = 500
        else: #if self.__geometry.layout == Layout.HOLES_8:
            self.__max_number_actions = 800

        self.__first_episode = True

    # ========== Observation space ============================================
    
    def __get_observation(self):
        """
        Get the current observed state of the environment.

        Returns
        -------
        numpy.float32[6]
            Array of observed values (see class documentation)

        """
        # TODO Why round decimals?
        return np.array([
            round(self.__ball_position[0], 3),   # Round 3 decimals
            round(self.__ball_position [1], 3),
            round(self.__last_ball_position[0], 3),
            round(self.__last_ball_position[1], 3),
            self.__x_degree,
            self.__y_degree
        ], dtype=np.float32)

    # ========== Reset ========================================================

    def reset(self):
        """
        Reset the environment.

        Parameters
        ----------
        None

        Returns
        -------
        numpy.ndarray
            Observation.
        None
            Information (currently not used).

        """
        # Set start position
        # TODO I do not understand __first_episode. How initialized for first episode? Why not initialized the same way?
        if self.__first_episode == False:
            match self.__geometry.layout:
                case Layout.HOLES_0_VIRTUAL | Layout.HOLES_0:
                    #area_start = [-6.06, 6.06, -5.76, 5.76]        # Inner area, closer to center
                    area_start = [-13.06, 13.06, -10.76, 10.76]     # Outer area
                    self.__ball_start_position[0] = random.uniform(area_start[0], area_start[1])
                    self.__ball_start_position[1] = random.uniform(area_start[2], area_start[3])
                case Layout.HOLES_2:
                    startpoint = [-0.79, 9.86]
                    self.__ball_start_position[0] = startpoint[0] + random.uniform(-0.4, 0.4)
                    self.__ball_start_position[1] = startpoint[1] + random.uniform(-0.4, 0.4)
                case Layout.HOLES_8:
                    startpoints = [[-5.82, -5.23], [-12.6, -7.03], [-9.8, -1.49], [-3.8, 1.29], [-12.85, 3.92], [0.13, 10.53]]
                    start_index = random.randint(0, len(startpoints )-1)
                    self.__ball_start_position[0] = startpoints[start_index][0] + random.uniform(-0.4, 0.4)
                    self.__ball_start_position[1] = startpoints[start_index][1] + random.uniform(-0.4, 0.4)
                case _:
                    # TODO What about Layout.HOLES_2_VIRTUAL?
                    raise Exception(f'Layout not supported: {self.__geometry.layout}')
        else:
            self.__first_episode = False

        # Field rotation
        self.__x_degree = 0.0
        self.__y_degree = 0.0

        # Ball
        self.__ball_position = self.__ball_start_position
        self.__last_ball_position = self.__ball_start_position
        self.__ball_physics.reset(position=self.__ball_start_position)

        # Rendering (ball in hole became invisible)
        if self.__render_mode == '3D':
            self.__render()
            self.__render_3d.ball_visibility(True)

        # Reset reward object and action history counter
        self.__rewards_rules.reset()
        self.__number_actions = 0

        # Observation space
        self.__observation_space = self.__get_observation()

        return self.__observation_space, {}

    # ========== Rendering ====================================================

    def __render(self):
        """
        Render state.

        Parameters
        ----------
        None

        Returns
        -------
        None.

        """
        if self.__render_mode == '3D':
            # Field rotation
            self.__render_3d.rotate_to(self.__x_degree, self.__y_degree)
            
            # Ball visibility
            if self.__ball_physics.is_in_hole:
                self.__render_3d.ball_visibility(False)
                
            # Ball position
            self.__render_3d.move_ball(self.__ball_position[0], self.__ball_position[1])

    # =========== Rewards =====================================================
    
    def __is_near_hole(self):
        """
        Calculates whether the ball is near a hole.

        Parameters
        ----------
        None

        Returns
        -------
        boolean
            True, if the ball is near a hole

        """
        # TODO Replace math.dist() calls by math.dist(a, b) instead of using lists [x, y] for points
        pos_x = self.__ball_position[0]
        pos_y = self.__ball_position[1]

        for hole in self.__geometry.holes.data:
            hole_center = hole["pos"]
            if math.dist([pos_x, pos_y], [hole_center.x, hole_center.y]) < 1.4 * self.__geometry.holes.radius:
                return True
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
        observation : numpy.ndarray
            Observation after applying the action.
        reward : float
            Reward of applying the action.
        done : boolean
            True if the episode has ended, else False.
        truncated : boolean
            True if the episode is terminated due to too many actions being taken.
        info : dict
            Information: not used

        """
        # Original state (observation space before action)
        state = self.__get_observation()

        # Field's rotation angles before and after applying the action
        stop_x_degree = start_x_degree = self.__x_degree
        stop_y_degree = start_y_degree = self.__y_degree
        
        # Does action to set degree of one axis (x or y) lead to rotation?
        if action < self.num_actions_per_component:
            stop_x_degree = self.__action_to_angle_degree[action]
        else:
            stop_y_degree = self.__action_to_angle_degree[action - self.num_actions_per_component]
        is_rotate_field = (stop_x_degree != start_x_degree) or (stop_y_degree != start_y_degree)

        # Remember last position before doing action
        self.__last_ball_position = self.__ball_position
        
        # New ball position
        if is_rotate_field == False:
            # Action does not rotate the field
            x_rad = stop_x_degree * pi/180.0
            y_rad = stop_y_degree * pi/180.0
            self.__ball_position = self.__ball_physics.step(x_rad=x_rad, y_rad=y_rad, number_steps=self.__physics_steps_per_action)
        else:            
            # Action does rotate the field
            start_x_rad = start_x_degree * pi/180.0
            start_y_rad = start_y_degree * pi/180.0
            stop_x_rad = stop_x_degree * pi/180.0
            stop_y_rad = stop_y_degree * pi/180.0

            delta_x_rad = (stop_x_rad - start_x_rad)
            delta_y_rad = (stop_y_rad - start_y_rad)
            max_delta = -1 * pi/180.0
            if delta_x_rad > 0 or delta_y_rad > 0:
                max_delta = 1 * pi/180.0
            # TODO Discuss and refactor "deadtime"
            deadtime = 1
            
            for i in range(self.__physics_steps_per_action): #The game board rotation is adjusted to 10ms updates.
                if i <= deadtime: # 20ms is only calculated with the old field rotation.
                    x_rad = start_x_rad
                    y_rad = start_y_rad
                elif stop_x_degree != start_x_degree:
                    y_rad = stop_y_rad
                    if abs(delta_x_rad) <= (i-deadtime) * abs(max_delta):
                        x_rad = stop_x_rad
                    else:
                        x_rad = start_x_rad + (i-deadtime) * max_delta
                elif stop_y_degree != start_y_degree:
                    x_rad = stop_x_rad
                    if abs(delta_y_rad) <= (i-deadtime) * abs(max_delta):
                        y_rad = stop_y_rad
                    else:
                        y_rad = start_y_rad + (i-deadtime) * max_delta
                self.__ball_position = self.__ball_physics.step(x_rad=x_rad, y_rad=y_rad)

            """
            # old field rotation
            delta_x_rad = (stop_x_rad - start_x_rad) / self.__physics_steps_per_action
            delta_y_rad = (stop_y_rad - start_y_rad) / self.__physics_steps_per_action
            
            for i in range (self.__physics_steps_per_action):
            
                x_rad = start_x_rad + (i+1) * delta_x_rad
                y_rad = start_y_rad + (i+1) * delta_y_rad
                self.__ball_position = self.__ball_physics.step(x_rad=x_rad, y_rad=y_rad)
            """

        # Store field's final rotation
        self.__x_degree = stop_x_degree
        self.__y_degree = stop_y_degree
        
        # Rendering active (Wait until period between steps has passed and render)
        if self.__render_mode == '3D':
            timestamp = time.time()
            elapsed_time = timestamp - self.__last_action_timestamp_sec
            wait_time = float(max(self.__actions_dt - elapsed_time, 0))
            time.sleep(wait_time)
            self.__last_action_timestamp_sec = timestamp + wait_time
            self.__render()

        # Ball reached destination?
        is_at_destination_x = (self.__ball_position[0] > self.__destination_x[0]) and (self.__ball_position[0] < self.__destination_x[1])
        is_at_destination_y = (self.__ball_position[1] > self.__destination_y[0]) and (self.__ball_position[1] < self.__destination_y[1])
        is_at_destination = is_at_destination_x and is_at_destination_y

        # Ball has fallen into a hole?
        if self.__geometry.layout.number_holes > 0:
            is_in_hole = self.__ball_physics.is_in_hole
            is_near_hole = self.__is_near_hole()
        else:
            is_in_hole = False
            is_near_hole = False
            
        # Next state (new observation space)
        self.__observation_space = self.__get_observation()

        # Reward
        reward = self.__rewards_rules.step(
            state = state,
            action = action,
            next_state = self.__observation_space,
            is_near_hole = is_near_hole,
            is_in_hole = is_in_hole,
            is_at_destination = is_at_destination)
        
        # Episode completed or truncated (i.e., max. number actions applied)?
        done = (is_at_destination or is_in_hole) and (self.__geometry.layout.number_holes > 0)
        self.__number_actions += 1
        truncated = (self.__number_actions >= self.__max_number_actions)

        return self.__observation_space, reward, done, truncated, {}

# -----------------------------------------------------------------------------
# Main (sample)
# -----------------------------------------------------------------------------

if __name__ == '__main__':
    env = LabyrinthEnv(layout=Layout.HOLES_8, render_mode='3D')
    env.reset()

    """for action in [0, 4, 6, 6]:
        env.step(action)"""
    for action in range(5):
        env.step(1)
    for action in range(30):
        env.step(9)
