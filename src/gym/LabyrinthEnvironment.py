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
from LabyrinthRewardArea import LabyrinthRewardArea

class LabyrinthEnvironment(gym.Env):
    """
    Observation space:
    ------------------
    1. The xy-position of the ball (2 values, np.float32)
    2. The last xy-position of the ball (2 values, np.float32)
       (also possible xy-components of the ball's velocity vector)
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
        layout : string
            Layout of holes and walls as defined in LabyrinthGeometry.py.
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

        # Defines geometric dimensions for reward calculations and specific rewards
        self.__rewardarea = LabyrinthRewardArea(layout=layout)

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
        self.__render_mode = render_mode
        if self.__render_mode == '3D':
            self.__render_3d = LabyrinthRender3D(self.__geometry)
            self.__render()

        # Declare observation space (see class documentation above)
        self.__observation_space = spaces.Box(
            low=np.array([-np.inf, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf], dtype=np.float32),
            high=np.array([np.inf, np.inf, np.inf, np.inf, np.inf, np.inf], dtype=np.float32)
        )

        # Declare action space (see class documentation above)
        self.__action_to_angle_degree = [-1, -0.5, 0, 0.5, 1]
        if (self.__geometry.layout == '2 holes real'):
            self.__action_to_angle_degree = [-1.5, -1, -0.5, 0, 0.5, 1, 1.5]
        self.num_actions_per_component = len(self.__action_to_angle_degree)

        self.__firstepisode = True

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
        # set different start positions
        if (self.__geometry.layout == '0 holes' or self.__geometry.layout == '0 holes real') and not self.__firstepisode:
            #area_start = [-6.06, 6.06, -5.76, 5.76]  # inner area, closer to middel
            area_start = [-13.06, 13.06, -10.76, 10.76] # outer area
            self.__ball_start_position.x = random.uniform(area_start[0], area_start[1])
            self.__ball_start_position.y = random.uniform(area_start[2], area_start[3])
        elif (self.__geometry.layout == '2 holes real') and not self.__firstepisode:
            startpoint = [-0.79, 9.86]
            self.__ball_start_position.x = startpoint[0] + random.uniform(-0.4, 0.4)
            self.__ball_start_position.y = startpoint[1] + random.uniform(-0.4, 0.4)
        elif (self.__geometry.layout == '8 holes') and not self.__firstepisode:
            startpoints = [[-5.82, -5.23], [-12.6, -7.03], [-9.8, -1.49], [-3.8, 1.29], [-12.85, 3.92], [0.13, 10.53]]
            start_index = random.randint(0, len(startpoints )-1)
            self.__ball_start_position.x = startpoints[start_index][0] + random.uniform(-0.4, 0.4)
            self.__ball_start_position.y = startpoints[start_index][1] + random.uniform(-0.4, 0.4)

        self.__firstepisode = False

        # observation space
        self.__observation_space = np.array([
            round(self.__ball_start_position.x, 3),
            round(self.__ball_start_position.y, 3),
            #0.0,
            #0.0,
            round(self.__ball_start_position.x, 3),
            round(self.__ball_start_position.y, 3),
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
        if self.__render_mode == '3D':
            self.__render()
            self.__render_3d.ball_visibility(True)

        # Init counters
        self.__number_actions = 0  # action history counter
        self.__interim_reward_counter = 0 # interim reward counter for threshold reward

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
            if self.__ball_physics.is_ball_in_hole:
                self.__render_3d.ball_visibility(False)
                
            # Ball position
            self.__render_3d.move_ball(self.__ball_position.x, self.__ball_position.y)


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
            if x_min < self.__last_ball_position.x and x_max > self.__last_ball_position.x and y_min < self.__last_ball_position.y and y_max > self.__last_ball_position.y:
                self.__last_distance = (self.__last_ball_position.x - self.__rewardarea.target_points[self.__progress][0])** 2 + (self.__last_ball_position.y - self.__rewardarea.target_points[self.__progress][1]) ** 2
                self.__current_distance = (self.__ball_position.x - self.__rewardarea.target_points[self.__progress][0])** 2 + (self.__ball_position.y - self.__rewardarea.target_points[self.__progress][1]) ** 2

                if self.__last_distance - self.__current_distance > 0.075: #sufficient change otherwise the agent will trick itself into rewarding changes in the 4th decimal place even though the ball feels like it's stuck to a wall.
                    return True
                elif self.__last_distance - self.__current_distance > 0:
                    self.__right_direction = True
                    return False
                else:
                    return False
            self.__progress += 1

        return False

    # ========== threshold reward ===========================================
    def __threshold_reward(self):
        """
        Calculates whether a defined threshold is crossed.
        (Not used now, but can be used to try threshold rewards)

        Parameters
        ----------
        None

        Returns
        -------
        1, if the threshold was crossed in the correct direction.
        -1, if the threshold was crossed in the wrong direction.
        0, otherwise

        """
        if len(self.__rewardarea.threshold_rewards) <= self.__interim_reward_counter:
            return 0

        axis = self.__rewardarea.threshold_rewards[self.__interim_reward_counter][0]
        direction = self.__rewardarea.threshold_rewards[self.__interim_reward_counter][1]
        threshold = self.__rewardarea.threshold_rewards[self.__interim_reward_counter][2]
        range_min = self.__rewardarea.threshold_rewards[self.__interim_reward_counter][3]
        range_max = self.__rewardarea.threshold_rewards[self.__interim_reward_counter][4]
        if axis == 'x':
            if direction < 0:
                if self.__last_ball_position.x > threshold and self.__ball_position.x <= threshold and self.__ball_position.y > range_min and self.__ball_position.y < range_max:
                    self.__interim_reward_counter += 1  # right crossing direction
                    return 1
                elif self.__last_ball_position.x < threshold and self.__ball_position.x >= threshold and self.__ball_position.y > range_min and self.__ball_position.y < range_max:
                    self.__interim_reward_counter -= 1  # false crossing direction
                    return -1
            elif self.__last_ball_position.x < threshold and self.__ball_position.x >= threshold and self.__ball_position.y > range_min and self.__ball_position.y < range_max:
                self.__interim_reward_counter += 1  # right crossing direction
                return 1
            elif self.__last_ball_position.x > threshold and self.__ball_position.x <= threshold and self.__ball_position.y > range_min and self.__ball_position.y < range_max:
                self.__interim_reward_counter -= 1  # false crossing direction
                return -1
        else:  # axis == 'y'
            if direction < 0:
                if self.__last_ball_position.y > threshold and self.__ball_position.y <= threshold and self.__ball_position.x > range_min and self.__ball_position.x < range_max:
                    self.__interim_reward_counter += 1  # right crossing direction
                    return 1
                elif self.__last_ball_position.y < threshold and self.__ball_position.y >= threshold and self.__ball_position.x > range_min and self.__ball_position.x < range_max:
                    self.__interim_reward_counter -= 1  # false crossing direction
                    return -1
            elif self.__last_ball_position.y < threshold and self.__ball_position.y >= threshold and self.__ball_position.x > range_min and self.__ball_position.x < range_max:
                self.__interim_reward_counter += 1 # right crossing direction
                return 1
            elif self.__last_ball_position.y > threshold and self.__ball_position.y <= threshold and self.__ball_position.x > range_min and self.__ball_position.x < range_max:
                self.__interim_reward_counter -= 1 # false crossing direction
                return -1
        return 0

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
        """
        # Only needed if threshold reward is to be used (comparison of the circle before and after the action execution)
        self.__last_radius_progress = 1
        for num in radius:
            if self.__last_distance > radius[self.__last_radius_progress-1]**2:
                break
            self.__last_radius_progress +=1"""

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
        boolean
            True, if the ball is near a hole

        """
        pos_x = self.__ball_position.x
        pos_y = self.__ball_position.y

        for hole in self.__geometry.holes.data:
            hole_center = hole["pos"]
            if ((pos_x - hole_center.x) ** 2 + (pos_y - hole_center.y) ** 2) < (self.__geometry.holes.radius *1.4) **2:
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
            Information: used for progress information for the evaluation of game plate 8 holes

        """
        # Field's rotation angles before and after applying the action
        stop_x_degree = start_x_degree = self.__x_degree
        stop_y_degree = start_y_degree = self.__y_degree
        # rotate only one axes
        if action < self.num_actions_per_component: # x-axis
            stop_x_degree = self.__action_to_angle_degree[action]
        else: # y-axis
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
                self.__ball_position = self.__ball_physics.step(x_rad=x_rad, y_rad=y_rad)"""

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
            
        # Observation_space
        self.__observation_space = np.array([
            round(self.__ball_position.x, 3), #Runden auf 3 nachkommastellen
            round(self.__ball_position.y, 3),
            #round(self.__ball_physics.get_velocity().x, 3),
            #round(self.__ball_physics.get_velocity().y, 3),
            round(self.__last_ball_position.x, 3),
            round(self.__last_ball_position.y, 3),
            self.__x_degree,
            self.__y_degree
        ], dtype=np.float32)

        # Ball reached destination?
        is_destination_x = (self.__ball_position.x > self.__destination_x[0]) and (self.__ball_position.x < self.__destination_x[1])
        is_destination_y = (self.__ball_position.y > self.__destination_y[0]) and (self.__ball_position.y < self.__destination_y[1])
        is_ball_at_destination = is_destination_x and is_destination_y

        # Ball has fallen into a hole?
        if self.__geometry.layout != '0 holes' and self.__geometry.layout != '0 holes real':
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
        self.__number_actions += 1 # Action history

        if self.__number_actions >= 300 and (self.__geometry.layout == '0 holes' or self.__geometry.layout == '0 holes real'):
            truncated = True
        elif self.__number_actions >= 500 and (self.__geometry.layout == '2 holes' or self.__geometry.layout == '2 holes real'):
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
    env = LabyrinthEnvironment(layout='0 holes', render_mode='3D')
    env.reset()

    """for action in [0, 4, 6, 6]:
        env.step(action)"""
    for action in range(5):
        env.step(1)
    for action in range(30):
        env.step(9)
