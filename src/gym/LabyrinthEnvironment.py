"""
BRIO labyrinth OpenAI gym environment.

The environment follows the gym documentation (last visited: 24.03.2024):
https://gymnasium.farama.org/tutorials/gymnasium_basics/environment_creation/

@authors: Sandra Lassahn, Marc Hensel
@contact: http://www.haw-hamburg.de/marc-hensel
@copyright: 2024
@version: 2024.05.14
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

    # TODO fps not used yet
    # TODO Apapt time period between steps so that it matches physics
    metadata = {'render_modes': ['3D'], 'render_fps': 10}

    # ========== Constructor ==================================================

    def __init__(self, layout, render_mode='3D'):
        """
        Constructor.

        Parameters
        ----------
        layout : string
            Layout of holes and walls as defined in LabyrinthGeometry.py. The default is '8 holes'
        render_mode : String, optional
            Render mode to visualize the states (or None). The default is '3D'.

        Returns
        -------
        None.

        """
        # Timing
        self.__time_steps_secs = 0.01
        
        # Create labyrinth geometry
        geometry = LabyrinthGeometry(layout=layout)

        # Rendering
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode
        if self.render_mode == '3D':
            self.__render_3d = LabyrinthRender3D(geometry)
        
        # Field rotation
        self.__x_degree = 0.0
        self.__y_degree = 0.0
                
        # Ball (physics, position, and destination)
        self.__ball_physics = LabyrinthBallPhysics(geometry=geometry, time_step_secs=self.__time_steps_secs)
        self.__ball_start_position = geometry.start_positions[layout]
        self.__ball_position = self.__ball_start_position
        self.__destination_x = geometry.destinations_xy[layout][0]
        self.__destination_y = geometry.destinations_xy[layout][1]

        # Declare observation space (see class documentation above)
        self.observation_space = spaces.Dict({
            'ball_position': spaces.Box(low=-np.inf, high=np.inf, shape=(2,), dtype=np.float32),
            'ball_velocity': spaces.Box(low=-np.inf, high=np.inf, shape=(2,), dtype=np.float32),
            'field_rotation': spaces.Box(low=-np.inf, high=np.inf, shape=(2,), dtype=np.float32)
        })

        # Declare action space (see class documentation above)
        # TODO Define action space differently
        self.action_space = spaces.Discrete(9)
        self.__action_to_angle_degree = {
            0: np.array([-0.8, -0.8]),
            1: np.array([0, -0.8]),
            2: np.array([0.8, -0.8]),
            3: np.array([-0.8, 0]),
            4: np.array([0, 0]),
            5: np.array([0.8, 0]),
            6: np.array([-0.8, 0.8]),
            7: np.array([0, 0.8]),
            8: np.array([0.8, 0.8])
        }

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

        # Action history and observation space
        self.last_action = 0
        self.number_actions = 0
        self.observation_space = {
            'ball_position': np.array([self.__ball_start_position.x, self.__ball_start_position.y]),
            'ball_velocity': np.array([0.0, 0.0]),
            'field_rotation': np.array([0.0, 0.0])
        }

        # Field rotation
        self.__x_degree = 0.0
        self.__y_degree = 0.0

        # Ball
        self.__ball_position = self.__ball_physics.calc_move(x_rad=0.0, y_rad=0.0, position=self.__ball_start_position)

        # Rendering (ball in hole became invisible)
        if self.render_mode == '3D':
            self.render()
            self.__render_3d.ball_visibility(True)

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
            if self.__ball_physics.game_state == -1:
                self.__render_3d.ball_visibility(False)
                
            # Ball position
            # TODO Check rotation of ball using x_rad and y_rad
            # TODO Adapt to frame rate or remove sleep()?
            x_rad = self.__x_degree * pi/180.0
            y_rad = self.__y_degree * pi/180.0
            self.__render_3d.move_ball(self.__ball_position.x, self.__ball_position.y, x_rad=x_rad, y_rad=y_rad)
            time.sleep(0.1)

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
        # Apply action to field's rotation
        self.__x_degree += float(self.__action_to_angle_degree[action][0])
        self.__y_degree += float(self.__action_to_angle_degree[action][1])

        # New ball position
        x_rad = float(self.__x_degree) * pi/180.0
        y_rad = float(self.__y_degree) * pi/180.0
        self.__ball_position = self.__ball_physics.calc_move(x_rad, y_rad)

        # Observation_space
        self.observation_space = {
            'ball_position': np.array([self.__ball_position.x, self.__ball_position.y]),
            'ball_velocity': np.array([self.__ball_physics.get_velocity().x, self.__ball_physics.get_velocity().y]),
            'field_rotation': np.array([self.__x_degree, self.__y_degree])
        }

        # Ball reached destination?
        is_destination_x = (self.__ball_position.x > self.__destination_x[0]) and (self.__ball_position.x < self.__destination_x[1])
        is_destination_y = (self.__ball_position.y > self.__destination_y[0]) and (self.__ball_position.y < self.__destination_y[1])
        is_ball_at_destination = is_destination_x and is_destination_y

        # Ball has fallen into a hole?
        is_ball_in_hole = (self.__ball_physics.game_state == -1)

        # Reward
        if is_ball_at_destination:
            print("Ball reached destination")
            reward = 500000
        elif is_ball_in_hole:
            print("Ball lost")
            reward = -1000000
        else:
            reward = -1

        # Episode completed?
        done = is_ball_at_destination or is_ball_in_hole

        # Action history
        # TODO Not used. Remove?
        self.last_action = action
        self.number_actions += 1

        return self.observation_space, reward, done, {}

# -----------------------------------------------------------------------------
# Main (sample)
# -----------------------------------------------------------------------------

if __name__ == '__main__':
    env = LabyrinthEnvironment(layout='8 holes', render_mode='3D')
    env.reset()

    for action in [2,6,3,1,0,5,5,5,5,5,5,5,5,4,4,4,4,4,4,4,4,4,4,4,4,4,4]:
        env.step(action)
        env.render()
