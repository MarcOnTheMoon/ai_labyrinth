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
import tkinter as tk
from tkinter import messagebox

import os
import sys

from App import App
from ServoCommunication import ServoCommunication

project_dir = os.path.dirname(os.path.abspath(__file__))
gym_dir = os.path.join(project_dir, '../../gym')
sys.path.append(gym_dir)
from LabyrinthGeometry import LabyrinthGeometry
from LabyrinthRewardArea import LabyrinthRewardArea

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
    there are 5 to 7 components per axis.
    """

    # ========== Constructor ==================================================

    def __init__(self, layout, cameraID = 0, actions_dt=0.1):
        """
        Constructor.
        
        For instance:
            actions_dt=0.1 means that an action is taken every 100 ms.
        

        Parameters
        ----------
        layout : string
            Layout of holes and walls as defined in LabyrinthGeometry.py. The default is '8 holes'
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

        # definition labyrinth geometry
        self.__geometry = LabyrinthGeometry(layout=layout)

        # Defines geometric dimensions for reward calculations and specific rewards
        self.__rewardarea = LabyrinthRewardArea(layout=layout)

        # definition of servocommunication
        self.__servo = ServoCommunication()

        # Field rotation to start
        self.__x_degree = 0.0
        self.__y_degree = 0.0
        self.__servo.rotate_to_angle(x_degree=self.__x_degree)
        time.sleep(0.1)  # warten zwischen zwei Datenübertragungen
        self.__servo.rotate_to_angle(y_degree=self.__y_degree)

        # definition camera settings and processing and calibration
        root = tk.Tk()
        root.withdraw()  # Hides the main window, not needed
        messagebox.showinfo("Kalibrationshinweis", "Klicken Sie nachfolgend die Spielfeldecken an und stellen Sie die Filterschwelle mit dem Schieberegler ein. Drücken Sie anschließend die ESC Taste.", icon='info')  # showing Message Box
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
        self.__action_to_angle_degree = [-1.0, -0.5, 0, 0.5, 1.0]
        if (self.__geometry.layout == '2 holes real'):
            self.__action_to_angle_degree = [-1.5, -1, -0.5, 0, 0.5, 1, 1.5]
        self.num_actions_per_component = len(self.__action_to_angle_degree)  # There are 9 possible actions per component (x,y)

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
        self.__number_actions = 0 # action history counter

        # Field rotation
        self.__x_degree = 0.0
        self.__y_degree = 0.0
        self.__servo.rotate_to_angle(x_degree=self.__x_degree)
        time.sleep(0.1)
        self.__servo.rotate_to_angle(y_degree=self.__y_degree)

        # Confirmation that the ball is on the start position
        root = tk.Tk()
        root.withdraw()
        messagebox.showinfo("Episodenstart", "Liegt der Ball an der Startposition?", icon='question') # showing Message Box
        self.__last_action_timestamp_sec = time.time()

        # Get and process next camera frame
        _ , ballCenter, _ = self.__app.processNextFrame()
        start_position = ballCenter
        # Convert ball position to the correct coordinate system
        self.__ball_start_position[0] = start_position[0] * self.__position_factor[0] - self.__geometry.field.size_x / 2
        self.__ball_start_position[1] = -start_position[1] * self.__position_factor[1] + self.__geometry.field.size_y / 2
        print (self.__ball_start_position)
        # observation space
        self.__observation_space = np.array([
            round(self.__ball_start_position[0], 3),
            round(self.__ball_start_position[1], 3),
            round(self.__ball_start_position[0], 3),
            round(self.__ball_start_position[1], 3),
            self.__x_degree,
            self.__y_degree
        ], dtype=np.float32)

        # Ball
        self.__ball_position = self.__ball_start_position
        self.__last_ball_position = self.__ball_start_position

        return self.__observation_space, {}

    #=========== right direction reward =======================================
    def __interim_reward(self):
        """
        Calculation to determine if the ball moved in the correct direction.

        Parameters
        ----------
        None

        Returns
        -------
        boolean
            True, if the ball has made a larger positional change in the correct maze direction

        """
        self.__progress = 0 # Progress describes the labyrintharea advancement, with numbering increasing from the goal to start, because the areas are defined from the goal to the start.
        self.__right_direction = False # __right_direction change to True when the ball made a smaller positional change in the correct maze direction
        for x_min, x_max, y_min, y_max in self.__rewardarea.areas:
            if x_min < self.__last_ball_position[0] and x_max > self.__last_ball_position[0] and y_min < self.__last_ball_position[1] and y_max > self.__last_ball_position[1]:
                self.__last_distance = (self.__last_ball_position[0] - self.__rewardarea.target_points[self.__progress][0])** 2 + (self.__last_ball_position[1] - self.__rewardarea.target_points[self.__progress][1]) ** 2
                self.__current_distance = (self.__ball_position[0] - self.__rewardarea.target_points[self.__progress][0])** 2 + (self.__ball_position[1] - self.__rewardarea.target_points[self.__progress][1]) ** 2

                if self.__last_distance - self.__current_distance > 0.075: #sufficient change otherwise the agent will trick itself into rewarding changes in the 4th decimal place even though the ball feels like it's stuck to a wall.
                    return True
                elif self.__last_distance - self.__current_distance > 0:
                    self.__right_direction = True
                    return False
                else:
                    return False
            self.__progress += 1

        return False

    # ========== 0hole reward ===========================================
    def __zerohole_reward(self):
        """
        calculates in witch circular segment the ball is lacated only used for layout "0 holes" and "0 holes real"

        Parameters
        ----------
        None

        Returns
        -------
        radius_progress: int
            The higher the number, the closer the ball is to the center
        """
        if self.__geometry.layout == '0 holes real':
            radius = [10, 7.5, 5, 2.5, 1.25]
        elif self.__geometry.layout == '0 holes':
            radius = [5, 2.5, 1.25]
        radius_progress = 1

        for num in radius:
            if self.__current_distance > radius[radius_progress-1]**2:
                return radius_progress
            radius_progress +=1
        return radius_progress

    # ========== close_hole_discount reward ===================================
    def __close_hole_discount(self):
        """
        Calculates whether the ball is near a hole.

        Parameters
        ----------
        None

        Returns
        -------
        Boolean
            True, if the ball is near a hole

        """
        pos_x = self.__ball_position[0]
        pos_y = self.__ball_position[1]

        for hole in self.__geometry.holes.data:
            hole_center = hole["pos"]
            if ((pos_x - hole_center.x) ** 2 + (pos_y - hole_center.y) ** 2) < (self.__geometry.holes.radius * 1.4) **2:
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
            Information: used for progress information for the evaluation of game plate 8 holes in simulation

        """
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

        #Ballposition
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
        # Observation_space
        self.__observation_space = np.array([
            round(self.__ball_position[0], 3),
            round(self.__ball_position[1], 3),
            round(self.__last_ball_position[0], 3),
            round(self.__last_ball_position[1], 3),
            self.__x_degree,
            self.__y_degree
        ], dtype=np.float32)

        # Ball reached destination?
        is_destination_x = (self.__ball_position[0] > self.__destination_x[0]) and (self.__ball_position[0] < self.__destination_x[1])
        is_destination_y = (self.__ball_position[1] > self.__destination_y[0]) and (self.__ball_position[1] < self.__destination_y[1])
        is_ball_at_destination = is_destination_x and is_destination_y

        # Ball has fallen into a hole?
        if self.__geometry.layout != '0 holes' and self.__geometry.layout != '0 holes real':
            # TODO aus Kamera bestimmen, ball_physics doesnt exist
            is_ball_in_hole = self.__ball_physics.is_ball_in_hole
            is_ball_to_close_hole = self.__close_hole_discount()
        else:
            is_ball_in_hole = False
            is_ball_to_close_hole = False

        # Reward
        if self.__geometry.layout == '8 holes' or self.__geometry.layout == '2 holes' or self.__geometry.layout == '2 holes real':
            if is_ball_at_destination:
                reward = self.__rewardarea.reward_dict['is_ball_at_destination']
                print("Ball reached destination")
            elif is_ball_in_hole:
                reward = self.__rewardarea.reward_dict['is_ball_in_hole']
                print("Ball lost")
            elif is_ball_to_close_hole:
                reward = self.__rewardarea.reward_dict['is_ball_to_close_hole']
                print("Ball close to hole")
            elif self.__interim_reward():
                reward = self.__rewardarea.reward_dict['interim_reward'](self.__progress, self.__rewardarea.areas)
            elif self.__right_direction:
                reward = self.__rewardarea.reward_dict['right_direction']
            else:
                reward = self.__rewardarea.reward_dict['default']
        elif self.__geometry.layout == '0 holes' or self.__geometry.layout == '0 holes real':
            interim_reward = self.__interim_reward()
            progress = self.__zerohole_reward()
            if is_ball_at_destination:
                print("Ball reached destination")
                reward = self.__rewardarea.reward_dict['destination']
            elif interim_reward or self.__right_direction:
                reward = self.__rewardarea.reward_dict['interim'].get(progress, self.__rewardarea.reward_dict['interim']['default'])
                if reward == 100:
                    print("Ball is close to center")
            else:
                reward = self.__rewardarea.reward_dict['default'].get(progress, self.__rewardarea.reward_dict['default']['default'])

        # Episode completed or truncated?
        done = (is_ball_at_destination or is_ball_in_hole) and self.__geometry.layout != '0 holes' and self.__geometry.layout != '0 holes real'
        self.__number_actions += 1  # Action history

        if self.__number_actions >= 300 and (
                self.__geometry.layout == '0 holes' or self.__geometry.layout == '0 holes real'):
            truncated = True
        elif self.__number_actions >= 500 and (
                self.__geometry.layout == '2 holes' or self.__geometry.layout == '2 holes real'):
            truncated = True
        elif self.__number_actions >= 800 and self.__geometry.layout == '8 holes':
            truncated = True
        else:
            truncated = False
        return self.__observation_space, reward, done, truncated, self.__progress


# -----------------------------------------------------------------------------
# Main (sample)
# -----------------------------------------------------------------------------

if __name__ == '__main__':
    env = LabyrinthMachine(layout='0 holes real')
    env.reset()

    for action in [0, 1, 6, 6]:
        env.step(action)
    for action in range(4):
        env.step(6)
    for action in range(20):
        env.step(9)
