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
import numpy as np
import time
import math
import tkinter as tk
from tkinter import messagebox

import os
import sys

from App import App
from ServoCommunication import ServoCommunication

project_dir = os.path.dirname(os.path.abspath(__file__))
virtual_dir = os.path.join(project_dir, '../../virtual')
sys.path.append(virtual_dir)
from LabLayouts import Layout, Geometry
from LabRewards import RewardsByAreas

class LabyrinthMachine():
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
    there are 5 components per axis.
    """

    # ========== Constructor ==================================================

    def __init__(self, layout, cameraID = 0, actions_dt=0.1):
        """
        Constructor.
        
        The parameters include one time period actions_dt, which specifies
        the frequency with which a reinforcement agent shall take actions.

        For instance:
            actions_dt=0.1 means that an action is taken every 100 ms.
        

        Parameters
        ----------
        layout :  enum Layout
            Layout of holes and walls as defined in LabLayouts.py.
        cameraID: int, optional
            ID of the camera
        actions_dt : float, optional
            Time period between steps (i.e., actions taken) [s]. The default is 0.1.

        Returns
        -------
        None.

        """
        # Timing
        self.__actions_dt = actions_dt
        self.__last_action_timestamp_sec = 0.0

        # Create labyrinth geometry
        assert type(layout) == Layout
        self.__geometry = Geometry(layout=layout)

        # Object to determine rewards of an step
        self.__rewards_rules = RewardsByAreas(layout=layout)

        # definition of servocommunication
        self.__servo = ServoCommunication()

        # Field rotation to start
        self.__x_degree = 0.0
        self.__y_degree = 0.0
        self.__servo.rotate_to_angle(x_degree=self.__x_degree)
        time.sleep(0.1)  # wait between two data transfers
        self.__servo.rotate_to_angle(y_degree=self.__y_degree)

        # definition camera settings and processing and calibration
        root = tk.Tk()
        root.withdraw()  # Hides the main window, not needed
        messagebox.showinfo("Calibration note", "Click on the corners of the playing field below and adjust the filter threshold using the slider. Then press the ESC key.", icon='info')  # showing Message Box
        self.__app = App(cameraID=cameraID, isShowImageAnalysis=True)
        self.__app.run() # only once for calibration

        # Converting factor for translating camera pixels to real-world dimensions
        camera_field_x, camera_field_y = self.__app.imaging.acquisition.get_field()
        self.__position_factor = [(self.__geometry.field.size_x / camera_field_x), (self.__geometry.field.size_y / camera_field_y)]
                
        # init Ballparameters (destination)
        self.__ball_start_position = [0.0, 0.0]
        self.__destination_x = self.__geometry.destinations_xy[layout][0]
        self.__destination_y = self.__geometry.destinations_xy[layout][1]

        # Declare action space (see class documentation above)
        self.__action_to_angle_degree = np.array([-1, -0.5, 0, 0.5, 1], dtype=np.float32) # maybe increase the angles of the actionspace
        self.num_actions_per_component = len(self.__action_to_angle_degree) # There are 5 possible actions per component (x,y)

        # defines max actions per episode for truncated
        if self.__geometry.layout.number_holes == 0:
            self.__max_number_actions = 300
        else:
            self.__max_number_actions = 800

    # ========== Observation space ============================================

    def __get_observation(self):
        """
        Get the current observed state of the environment.

        Returns
        -------
        numpy.float32[6]
            Array of observed values (see class documentation)

        """
        return np.array([
            self.__ball_position[0],
            self.__ball_position[1],
            self.__last_ball_position[0],
            self.__last_ball_position[1],
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
        # Field rotation
        self.__x_degree = 0.0
        self.__y_degree = 0.0
        self.__servo.rotate_to_angle(x_degree=self.__x_degree)
        time.sleep(0.1)
        self.__servo.rotate_to_angle(y_degree=self.__y_degree)

        # Confirmation that the ball is on the start position
        root = tk.Tk()
        root.withdraw()
        messagebox.showinfo("Episode start", "Is the ball at the starting position?", icon='question') # showing Message Box
        self.__last_action_timestamp_sec = time.time()

        # Get and process next camera frame
        _ , ballCenter, _ = self.__app.processNextFrame()
        start_position = ballCenter
        # Convert ball position to the correct coordinate system
        self.__ball_start_position[0] = start_position[0] * self.__position_factor[0] - self.__geometry.field.size_x / 2
        self.__ball_start_position[1] = -start_position[1] * self.__position_factor[1] + self.__geometry.field.size_y / 2
        print (self.__ball_start_position)

        # Ball
        self.__ball_position = self.__ball_start_position
        self.__last_ball_position = self.__ball_start_position

        # Reset reward object and action history counter
        self.__rewards_rules.reset()
        self.__number_actions = 0

        # Observation space
        self.__observation_space = self.__get_observation()

        return self.__observation_space, {}

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
        info : int
            Information: not used

        """
        # Original state (observation space before action)
        state = self.__get_observation()
        
        # Field's rotation angles before and after applying the action
        stop_x_degree = start_x_degree = self.__x_degree
        stop_y_degree = start_y_degree = self.__y_degree
        # rotate only one axes
        if action < self.num_actions_per_component:
            stop_x_degree = self.__action_to_angle_degree[action]
            if stop_x_degree != start_x_degree:
                self.__servo.rotate_to_angle(x_degree = stop_x_degree)
        else:
            stop_y_degree = self.__action_to_angle_degree[action - self.num_actions_per_component]
            if stop_y_degree != start_y_degree:
                self.__servo.rotate_to_angle(y_degree = stop_y_degree)

        # Store field's final rotation
        self.__x_degree = stop_x_degree
        self.__y_degree = stop_y_degree

        # Remember last position before doing action
        self.__last_ball_position = self.__ball_position

        # (Wait until period between steps has passed)
        timestamp = time.time()
        elapsed_time = timestamp - self.__last_action_timestamp_sec
        wait_time = float(max(self.__actions_dt - elapsed_time, 0))
        print(f'waittim {wait_time}')
        time.sleep(wait_time)
        self.__last_action_timestamp_sec = time.time()

        # Get and process next camera frame
        _, ballCenter, _ = self.__app.processNextFrame()
        if ballCenter is not None:
            start_position = ballCenter
            # Ballposition ins richtige Koordinatensystem umrechnen
            self.__ball_start_position[0] = start_position[0] * self.__position_factor[0] - self.__geometry.field.size_x / 2
            self.__ball_start_position[1] = - start_position[1] * self.__position_factor[1] + self.__geometry.field.size_y / 2
        else:
            print("No new ball position could be determined, using last ballposition")
        print(self.__ball_start_position)
        # Next state (new observation space)
        self.__observation_space = self.__get_observation()

        # Ball reached destination?
        is_at_destination_x = (self.__ball_position[0] > self.__destination_x[0]) and (self.__ball_position[0] < self.__destination_x[1])
        is_at_destination_y = (self.__ball_position[1] > self.__destination_y[0]) and (self.__ball_position[1] < self.__destination_y[1])
        is_at_destination = is_at_destination_x and is_at_destination_y

        # Ball has fallen into a hole?
        if self.__geometry.layout.number_holes > 0:
            # TODO aus Kamera bestimmen, ball_physics doesnt exist. so its initialized with false
            #is_in_hole = self.__ball_physics.is_in_hole
            #is_near_hole = self.__is_near_hole()
            is_in_hole = False
            is_near_hole = False
        else:
            is_in_hole = False
            is_near_hole = False

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
    env = LabyrinthMachine(layout=Layout.HOLES_0)
    env.reset()

    for action in [0, 1, 6, 6]:
        env.step(action)
    for action in range(4):
        env.step(6)
    for action in range(20):
        env.step(9)
