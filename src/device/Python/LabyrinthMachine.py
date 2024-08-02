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

class LabyrinthMachine(gym.Env):
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
        self.layout = layout

        # definition of aeward areas and targetpoints
        self.__rewardarea = LabyrinthRewardArea(layout=layout)

        # definition camera settings and processing
        root = tk.Tk()
        root.withdraw()  # Versteckt das Hauptfenster, da wir es nicht benötigen
        messagebox.showinfo("Kalibrationshinweis", "Klicken Sie nachfolgend die Spielfeldecken an und stellen Sie die Filterschwelle mit dem Schieberegler ein. Drücken Sie anschließend die ESC Taste.", icon='info')  # Message Box anzeigen
        self.__app = App(cameraID=cameraID, isShowImageAnalysis=True)
        self.__app.run()

        #Umrechnungsfactor für Kamerapixelposition in reale Position überführen
        #imageAcquisition = ImageAcquisition()
        camera_field_x, camera_field_y = self.__app.imaging.acquisition.get_field()
        self.__position_factor_x = self.__geometry.field.size_x / camera_field_x
        self.__position_factor_y = self.__geometry.field.size_y / camera_field_y
        # definition of servocommunication
        self.__servo = ServoCommunication()

        
        # Field rotation
        self.__x_degree = 0.0
        self.__y_degree = 0.0
        self.__servo.rotate_to_angle(x_degree=self.__x_degree)
        time.sleep(0.1) # warten zwischen zwei Datenübertragungen
        self.__servo.rotate_to_angle(y_degree=self.__y_degree)
                
        # Ball (destination)
        self.__ball_start_position = [0.0, 0.0]
        self.__destination_x = self.__geometry.destinations_xy[layout][0]
        self.__destination_y = self.__geometry.destinations_xy[layout][1]

        # Declare observation space (see class documentation above)
        self.observation_space = spaces.Box(
            low=np.array([-np.inf, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf], dtype=np.float32),
            high=np.array([np.inf, np.inf, np.inf, np.inf, np.inf, np.inf], dtype=np.float32)
        )

        # Declare action space (see class documentation above)
        self.__action_to_angle_degree = [-1.0, -0.5, 0, 0.5, 1.0]
        if (self.layout == '2 holes real'):
            self.__action_to_angle_degree = [-1.5, -1, -0.5, 0, 0.5, 1, 1.5]
        self.num_actions_per_component = len(self.__action_to_angle_degree)  # There are 9 possible actions per component (x,y)
        self.action_space = spaces.Discrete(2 * self.num_actions_per_component)

    # ========== Reset ========================================================

    def reset(self):
        """
        Reset the environment.

        Parameters
        ----------
        None

        Returns
        -------
        Dict
            Observation.
        None
            Information (currently not used).

        """
        self.number_actions = 0 # action history counter

        # Field rotation
        self.__x_degree = 0.0
        self.__y_degree = 0.0
        self.__servo.rotate_to_angle(x_degree=self.__x_degree)
        time.sleep(0.1)
        self.__servo.rotate_to_angle(y_degree=self.__y_degree)

        #bestätigung Ball liegt auf der Startposition
        root = tk.Tk()
        root.withdraw()  # Versteckt das Hauptfenster, da wir es nicht benötigen
        messagebox.showinfo("Episodenstart", "Liegt der Ball an der Startposition?", icon='question') # Message Box anzeigen
        self.__last_action_timestamp_sec = time.time()

        # Get and process next camera frame
        _ , ballCenter, _ = self.__app.processNextFrame()
        start_position = ballCenter
        # Ballposition ins richtige Koordinatensystem umrechnen
        self.__ball_start_position[0] = start_position[0] * self.__position_factor_x - self.__geometry.field.size_x / 2
        self.__ball_start_position[1] = -start_position[1] * self.__position_factor_y + self.__geometry.field.size_y / 2
        print (self.__ball_start_position)
        # observation space
        self.observation_space = np.array([
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


        return self.observation_space, {}

    #=========== right direction reward =======================================
    def interim_reward(self):
        """
        Calculation to determine if the ball moved in the correct direction.

        Returns
        -------
        True, if the ball has made a larger positional change in the correct maze direction

        """
        self.__progress = 0
        self.__right_direction = False
        for x_min, x_max, y_min, y_max in self.__rewardarea.areas:
            if x_min < self.__last_ball_position[0] and x_max > self.__last_ball_position[0] and y_min < self.__last_ball_position[1] and y_max > self.__last_ball_position[1]:
                self.__last_distance = (self.__last_ball_position[0] - self.__rewardarea.target_points[self.__progress][0])** 2 + (self.__last_ball_position[1] - self.__rewardarea.target_points[self.__progress][1]) ** 2
                self.__current_distance = (self.__ball_position[0] - self.__rewardarea.target_points[self.__progress][0])** 2 + (self.__ball_position[1] - self.__rewardarea.target_points[self.__progress][1]) ** 2

                if self.__last_distance - self.__current_distance > 0.075: #genügend Änderung sonst tricks der agent ein aus, dass Änderungen in der 4 Nachkommastelle belohnt werden obwohl der ball gefühlt an einer Wand klebt
                    return True #in die richtige Richtung
                elif self.__last_distance - self.__current_distance > 0:
                    self.__right_direction = True
                    return False
                else:
                    return False
            self.__progress += 1

        return False

    # ========== 0hole reward ===========================================
    def zerohole_reward(self):
        """
        calculates in witch circular segment the ball is lacated

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
    def close_hole_discount(self):
        pos_x = self.__ball_position[0]
        pos_y = self.__ball_position[1]

        for hole in self.__geometry.holes.data:
            hole_center = hole["pos"]
            if ((pos_x - hole_center.x) ** 2 + (pos_y - hole_center.y) ** 2) < (self.__geometry.holes.radius + self.__geometry.holes.radius*0.4) **2:
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
        observation :
            Observation after applying the action.
        reward : int
            Reward of applying the action.
        done : boolean
            True if the episode has ended, else False.
        truncated : boolean
            True if the episode ist truncated
        info : dict
            Information: umschaltschwelle für das evaluieren mehrerer zusammengesetzer Netze für ein Spiel.

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
        if ballCenter is None:
            time.sleep(0.005)
            _, ballCenter, _ = self.__app.processNextFrame()
            if ballCenter is None:
                root = tk.Tk()
                root.withdraw()  # Versteckt das Hauptfenster, da wir es nicht benötigen
                messagebox.showinfo("Kugelerkennungsproblem",
                                    "Es konnte wiederholt keine Kugelposition ermittelt werden.", icon='warning')  # Message Box anzeigen
                _, ballCenter, _ = self.__app.processNextFrame()
        start_position = ballCenter
        # Ballposition ins richtige Koordinatensystem umrechnen
        self.__ball_start_position[0] = start_position[0] * self.__position_factor_x - self.__geometry.field.size_x / 2
        self.__ball_start_position[1] = - start_position[1] * self.__position_factor_y + self.__geometry.field.size_y / 2
        print(self.__ball_start_position)
        # Observation_space
        self.observation_space = np.array([
            round(self.__ball_position[0], 3), #Runden auf 3 nachkommastellen
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
        if self.layout != '0 holes' and self.layout != '0 holes real':
            # TODO aus Kamera bestimmen
            is_ball_in_hole = self.__ball_physics.is_ball_in_hole
            is_ball_to_close_hole = self.close_hole_discount()
        else:
            is_ball_in_hole = False
            is_ball_to_close_hole = False

        # Reward
        if self.__geometry.layout == '8 holes':
            if is_ball_at_destination:
                print("Ball reached destination")
                reward = 1000
            elif is_ball_in_hole:
                print("Ball lost")
                reward = -300
            elif is_ball_to_close_hole:
                print("Ball close to hole")
                reward = -15
            elif self.interim_reward():
                reward = 6 / len(self.__rewardarea.areas) * (
                            len(self.__rewardarea.areas) - self.__progress)  # Positively reward progress, giving more reward for each additional tile in the correct movement direction.
            elif self.__right_direction:
                reward = -1
            else:
                reward = -2
        elif self.__geometry.layout == '2 holes':
            if is_ball_at_destination:
                print("Ball reached destination")
                reward = 600
            elif is_ball_in_hole:
                print("Ball lost")
                reward = -200
            elif is_ball_to_close_hole:
                print("Ball close to hole")
                reward = -1
            elif self.interim_reward():
                reward = 3 / len(self.__rewardarea.areas) * (
                            len(self.__rewardarea.areas) - self.__progress)  # Positively reward progress, giving more reward for each additional tile in the correct movement direction.
            elif self.__right_direction:
                reward = -1
            else:
                reward = -2
        elif self.__geometry.layout == '2 holes real':
            if is_ball_at_destination:
                print("Ball reached destination")
                reward = 600
            elif is_ball_in_hole:
                print("Ball lost")
                reward = -200
            elif is_ball_to_close_hole:
                print("Ball close to hole")
                reward = -10
            elif self.interim_reward():
                reward = 3 / len(self.__rewardarea.areas) * (
                            len(self.__rewardarea.areas) - self.__progress)  # Positively reward progress, giving more reward for each additional tile in the correct movement direction.
            elif self.__right_direction:
                reward = -1
            else:
                reward = -2
        elif self.__geometry.layout == '0 holes':
            if is_ball_at_destination:
                print("Ball reached destination")
                reward = 600
            elif self.interim_reward():
                if self.__current_distance < 1.25 ** 2:
                    reward = 100
                    print("close to destination_3")
                elif self.__current_distance < 2.5 ** 2:
                    reward = 20
                    print("close to destination_2")
                elif self.__current_distance < 5 ** 2:
                    reward = 2
                    print("close to destination_1")
                else:
                    reward = -0.2
                    # print("right direction")
            else:
                if self.__current_distance < 1.25 ** 2:
                    reward = -0.2
                    print("close to destination false direction")
                    if self.__current_distance < 2.5 ** 2:
                        reward = -0.4
                else:
                    reward = -1
        else:  # self.__geometry.layout == '0 holes real':
            self.interim_reward()
            progress = self.zerohole_reward()
            if is_ball_at_destination:
                print("Ball reached destination")
                reward = 600
            elif (self.interim_reward() or self.__right_direction) and progress != 1:
                if progress == 2:
                    reward = -0.4
                elif progress == 3:
                    reward = -0.2
                elif progress == 4:
                    reward = 2
                elif progress == 5:
                    reward = 25
                elif progress == 6:
                    reward = 100
            elif progress == 6:
                reward = -0.2
            elif progress == 5:
                reward = -0.4
            else:
                reward = -2

        # Episode completed or truncated?
        done = (is_ball_at_destination or is_ball_in_hole) and self.__geometry.layout != '0 holes' and self.__geometry.layout != '0 holes real'
        self.number_actions += 1  # Action history

        if self.number_actions >= 300 and (
                self.__geometry.layout == '0 holes' or self.__geometry.layout == '0 holes real'):
            truncated = True
        elif self.number_actions >= 500 and (
                self.__geometry.layout == '2 holes' or self.__geometry.layout == '2 holes real'):
            truncated = True
        elif self.number_actions >= 800 and self.__geometry.layout == '8 holes':
            truncated = True
        else:
            truncated = False
        return self.observation_space, reward, done, truncated, self.__progress


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
