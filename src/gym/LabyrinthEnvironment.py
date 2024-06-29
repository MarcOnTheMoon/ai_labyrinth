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
import random
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
        self.__geometry = LabyrinthGeometry(layout=layout)
        self.layout = layout
        
        # Field rotation
        self.__x_degree = 0.0
        self.__y_degree = 0.0
                
        # Ball (physics, position, and destination)
        self.__ball_physics = LabyrinthBallPhysics(geometry=self.__geometry, dt=self.__physics_dt)
        self.__ball_start_position = self.__geometry.start_positions[layout]
        self.__ball_position = self.__ball_start_position
        self.__destination_x = self.__geometry.destinations_xy[layout][0]
        self.__destination_y = self.__geometry.destinations_xy[layout][1]

        # Rendering
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode
        if self.render_mode == '3D':
            self.__render_3d = LabyrinthRender3D(self.__geometry)
            self.render()

        # Declare observation space (see class documentation above)
        self.observation_space = spaces.Box(
            low=np.array([-np.inf, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf], dtype=np.float32),
            high=np.array([np.inf, np.inf, np.inf, np.inf, np.inf, np.inf], dtype=np.float32)
        )


        # Declare action space (see class documentation above)
        #self.action_space = spaces.MultiDiscrete([num_actions_per_component, num_actions_per_component]) # MultiDiscrete Aktionsraum erlaubt es, für jede Komponente (x und y) unabhängige diskrete Werte zu definieren. Hier haben beide Komponenten 9 mögliche Werte.
        #self.__action_to_angle_degree = [-2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2]
        self.__action_to_angle_degree = [-1, -0.5, 0, 0.5, 1]
        if (self.__geometry.layout == '2 holes real'):
            self.__action_to_angle_degree = [-1.5, -1, -0.5, 0, 0.5, 1, 1.5]
        self.num_actions_per_component = len(self.__action_to_angle_degree)  # There are 9 possible actions per component (x,y)
        self.action_space = spaces.Discrete(2 * self.num_actions_per_component)

        self.firstepisode = True

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

        self.number_actions = 0 # action history counter

        # observation space
        if (self.__geometry.layout == '0 holes') and not self.firstepisode:
            self.__ball_start_position.x = random.uniform(self.__geometry.area_start[0], self.__geometry.area_start[1])
            self.__ball_start_position.y = random.uniform(self.__geometry.area_start[2], self.__geometry.area_start[3])
        elif (self.__geometry.layout == '2 holes real') and not self.firstepisode:
            startpoint = [-0.79, 9.86]
            self.__ball_start_position.x = startpoint[0] + random.uniform(-0.4, 0.4)
            self.__ball_start_position.y = startpoint[1] + random.uniform(-0.4, 0.4)
        elif (self.__geometry.layout == '8 holes') and not self.firstepisode:
            startpoints = [[4.54, 9.6], [-1.2, 1.14], [3.35, -2.99], [0.94, -5.23], [-5.82, -5.23], [-12.6, -7.03]]
            start_index = random.randint(0, len(startpoints )-1)
            self.__ball_start_position.x = startpoints[start_index][0] + random.uniform(-0.4, 0.4)
            self.__ball_start_position.y = startpoints[start_index][1] + random.uniform(-0.4, 0.4)

        self.firstepisode = False

        self.observation_space = np.array([
            round(self.__ball_start_position.x, 3),
            round(self.__ball_start_position.y, 3),
            0.0,
            0.0,
            #round(self.__ball_start_position.x, 3),
            #round(self.__ball_start_position.y, 3),
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
            self.__render_3d.move_ball(self.__ball_position.x, self.__ball_position.y)
            
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

        #Reihnfolge ziel nach anfang, mit jeweils [x,y] Zwischenzielkoordinaten
        if self.__geometry.layout == '0 holes':
            target_points = [[0, 0]]

        elif self.__geometry.layout == '2 holes':
            target_points = [[-0.13, -6],
                             [-0.13, -1.52],
                             [0.33, 1.2],
                             [0.33, 1.2],
                             [1.97, 5.81]]

        elif self.__geometry.layout == '2 holes real':
            target_points = [[1.24, -10.42],
                             [0.3, -6.77],
                             [-0.22, -3.57],
                             [1.06, -0.92],
                             [1.06, -0.92],

                             [2.71, 4.28],
                             [1.13, 6.55]]

        elif self.__geometry.layout == '8 holes':
            target_points = [[-4.4, -10.32],
                             [-4.4, -10.32],
                             [5.82, -10.32],
                             [7.33, -6.88],
                             [8.89, -8.29],

                             [12.54, -7.53],
                             [12.99, -5],
                             [10.35, -1.15],
                             [12.53, -1.51],
                             [12.41, 3.74],

                             [11.39, 5.36],
                             [11.43, 7.01],
                             [8.71, 10.19],
                             [4.54, 9.6],
                             [6.07, 7.55],

                             [5.52, 3.39],
                             [-1.2, 1.14],
                             [-1.22, -0.73],
                             [-0.05, -2.6],
                             [3.35, -2.99],

                             [0.94, -5.23],
                             [-2.74, -7.98],
                             [-5.82, -5.23],
                             [-8.1, -7.41],
                             [-12.58, -7.03],

                             [-13.16, -4.7],
                             [-11.63, -3.18],
                             [-9.8, -1.49],
                             [-5.55, -1.84],
                             [-3.31, 1.29],

                             [-6.09, 4.15],
                             [-9.51, 4.48],
                             [-11.05, 3.58],
                             [-12.85, 3.92],
                             [-12.45, 10.83],

                             [-12.45, 10.83]]
        # [x_min, x_max, y_min, y_max]
        if self.__geometry.layout == '0 holes':
            self.areas = [[-13.06, 13.06, -10.76, 10.76]]

        elif self.__geometry.layout == '2 holes':
            self.areas = [[-3.13, 3.33, -6.53, -1.03],
                          [-3.13, 1.0, -1.03, 4.08],
                          [1.0, 3.33, 0.47, 4.08],
                          [-0.7, 3.33, 4.08, 6.48],
                          [-3.13, 3.33, 6.48, 11.4]]

        elif self.__geometry.layout == '2 holes real':
            self.areas = [[-2.09, 4.6, -9.9, -6.05],
                          [-2.09, 1.96, -6.05, -2.92],
                          [-2.09, 1.96, -2.92, 2.28],
                          [1.96, 4.6, -2.92, 2.28],
                          [0.14, 4.6, 2.28, 5.03],

                          [0.14, 4.6, 5.03, 10.27],
                          [-2.09, 0.14, 5.03, 10.27]]

        elif self.__geometry.layout == '8 holes':
            self.areas = [[0.14, 7.46, -11.40, -9.52],
                          [-58.6, 0.14, -11.40, -9.52],
                          [5.21, 7.46, -9.52, -4.63],
                          [7.46, 9.46, -11.4, -4.63],
                          [9.46, 13.7, -11.4, -6.58],

                          [9.46, 13.7, -6.58, -3.98],
                          [9.46, 13.7, -3.98, -0.65],
                          [9.46, 13.7, -0.65, 1.9],
                          [9.46, 13.7, 1.9, 4.15],
                          [9.46, 13.7, 4.15, 5.75],

                          [9.46, 13.7, 5.75, 7.28],
                          [8.23, 13.7, 7.28, 11.4],
                          [2.26, 8.23, 9.09, 11.4],
                          [2.26, 7.83, 7.08, 9.09],
                          [4.81, 8.82, 2.16, 7.08],

                          [-1.98, 4.81, 0.6, 4.11],
                          [-1.98, 0.29, -1.23, 0.6],
                          [-1.98, 0.29, -3.56, -1.23],
                          [0.29, 4.59, -3.85, -1.23],
                          [0.16, 4.59, -6.95, -3.85],

                          [-3.38, 0.16, -8.86, -4.17],
                          [-6.26, -3.38, -8.86, -4.17],
                          [-8.88, -6.26, -11.4, -4.17],
                          [-13.7, -8.88, -11.4, -6.53],
                          [-13.7, -11.59, -6.53, -4.11],

                          [-13.7, -10.25, -4.11, -2.64],
                          [-12.1, -8.98, -2.64, -0.16],
                          [-8.98, -4.73, -3.53, 0.11],
                          [-4.73, -2.53, -3.53, 2.01],
                          [-6.58, -2.53, 2.01, 5.56],

                          [-10.13, -6.58, 2.39, 5.56],
                          [-11.69, -10.13, 0.61, 5.56],
                          [-13.7, -11.69, 0.61, 4.68],
                          [-13.7, -11.66, 4.68, 11.4],
                          [-11.66, -5.34, 9.99, 11.4],

                          [-5.34, 0.5, 9.99, 11.4]]

        self.__progress = 0
        self.__right_direction = False
        for x_min, x_max, y_min, y_max in self.areas:
            if x_min < self.__last_ball_position.x and x_max > self.__last_ball_position.x and y_min < self.__last_ball_position.y and y_max > self.__last_ball_position.y:
                self.__last_distance = (self.__last_ball_position.x - target_points[self.__progress][0])** 2 + (self.__last_ball_position.y - target_points[self.__progress][1]) ** 2
                self.__current_distance = (self.__ball_position.x - target_points[self.__progress][0])** 2 + (self.__ball_position.y - target_points[self.__progress][1]) ** 2

                if self.__last_distance - self.__current_distance > 0.075: #genügend Änderung sonst tricks der agent ein aus, dass Änderungen in der 4 Nachkommastelle belohnt werden obwohl der ball gefühlt an einer Wand klebt
                    return True #in die richtige Richtung
                else:
                    self.__right_direction = True
                    return False
            self.__progress += 1

        return False

    # ========== close_hole_discount reward ===================================
    def close_hole_discount(self):
        pos_x = self.__ball_position.x
        pos_y = self.__ball_position.y

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
        info : dict
            Information (currently not used).

        """
        # Field's rotation angles before and after applying the action
        stop_x_degree = start_x_degree = self.__x_degree
        stop_y_degree = start_y_degree = self.__y_degree
        # rotate only one axes
        if action < self.num_actions_per_component:
            stop_x_degree = self.__action_to_angle_degree[action]
        else:
            stop_y_degree = self.__action_to_angle_degree[action - self.num_actions_per_component]
        is_rotate_field = (stop_x_degree != start_x_degree) or (stop_y_degree != start_y_degree)

        self.__last_ball_position = self.__ball_position  # remember last position before step action
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
            round(self.__ball_position.x, 3), #Runden auf 3 nachkommastellen
            round(self.__ball_position.y, 3),
            round(self.__ball_physics.get_velocity().x, 3),
            round(self.__ball_physics.get_velocity().y, 3),
            #round(self.__last_ball_position.x, 3),
            #round(self.__last_ball_position.y, 3),
            self.__x_degree,
            self.__y_degree
        ], dtype=np.float32)

        # Ball reached destination?
        is_destination_x = (self.__ball_position.x > self.__destination_x[0]) and (self.__ball_position.x < self.__destination_x[1])
        is_destination_y = (self.__ball_position.y > self.__destination_y[0]) and (self.__ball_position.y < self.__destination_y[1])
        is_ball_at_destination = is_destination_x and is_destination_y

        # Ball has fallen into a hole?
        if self.__geometry.layout != '0 holes':
            is_ball_in_hole = self.__ball_physics.is_ball_in_hole
            is_ball_to_close_hole = self.close_hole_discount()
        else:
            is_ball_in_hole = False
            is_ball_to_close_hole = False

        # Reward
        if self.__geometry.layout == '8 holes':
            if is_ball_at_destination:
                print("Ball reached destination")
                reward = 800
            elif is_ball_in_hole:
                print("Ball lost")
                reward = -300
            elif is_ball_to_close_hole:
                print("Ball close to hole")
                reward = -10
            elif self.interim_reward():
                reward = 5/len(self.areas) *(len(self.areas)-self.__progress) #den wegfortschritt positiv belohnen, jede kachel weiter dann gibt es mehr Belohnung für die richtige Bewegungsrichtung
                #print("right direction")
            #elif self.__right_direction:
            #    reward = -1
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
                reward = -11
            elif self.interim_reward():
                #reward = 0.5
                reward = 2/len(self.areas) *(len(self.areas)-self.__progress) #den wegfortschritt positiv belohnen, jede kachel weiter dann gibt es mehr Belohnung für die richtige Bewegungsrichtung
                #print("right direction")
            else:
                reward = -1
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
                reward = 3/len(self.areas) *(len(self.areas)-self.__progress) #den wegfortschritt positiv belohnen, jede kachel weiter dann gibt es mehr Belohnung für die richtige Bewegungsrichtung
            elif self.__right_direction:
                reward = -1
            else:
                reward = -2
        else: #self.__geometry.layout == '0 holes'
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
                    #print("right direction")
            else:
                if self.__current_distance < 1.25 ** 2:
                    reward = -0.2
                    print("close to destination false direction")
                    if self.__current_distance < 2.5 ** 2:
                        reward = -0.4
                else:
                    reward = -1

        # Episode completed or truncated?
        done = (is_ball_at_destination or is_ball_in_hole) and self.__geometry.layout != '0 holes'
        self.number_actions += 1 # Action history

        if self.number_actions >= 300 and self.__geometry.layout == '0 holes':
            truncated = True
        elif self.number_actions >= 500 and (self.__geometry.layout == '2 holes' or self.__geometry.layout == '2 holes real'):
            truncated = True
        elif self.number_actions >= 800 and self.__geometry.layout == '8 holes':
            print(f'truncated at Ball Pos. x={self.__ball_position.x} y={self.__ball_position.y}')
            truncated = True
        else:
            truncated = False

        return self.observation_space, reward, done, truncated, {}


# -----------------------------------------------------------------------------
# Main (sample)
# -----------------------------------------------------------------------------

if __name__ == '__main__':
    env = LabyrinthEnvironment(layout='2 holes real', render_mode='3D')
    env.reset()

    for action in [1, 1, 6, 6]:
        env.step(action)
    for action in range(4):
        env.step(6)
    for action in range(20):
        env.step(9)
