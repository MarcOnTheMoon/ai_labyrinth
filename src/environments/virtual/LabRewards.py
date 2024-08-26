"""
Reward rules for a labyrinth OpenAI gym environment.

This file contains an abstract base class and classes implementin different
rules to determine rewards for steps of reinforcement learning:
    
    - RewardsByAreas: segment field into areas and move along a sequence of areas

@authors: Sandra Lassahn, Marc Hensel
@contact: http://www.haw-hamburg.de/marc-hensel
@copyright: 2024
@version: 2024.08.26
@license: CC BY-NC-SA 4.0, see https://creativecommons.org/licenses/by-nc-sa/4.0/deed.en
"""
import math
from abc import ABC, abstractmethod 
from LabLayouts import Layout

# TODO Replace python lists by numpy arrays for better performance?

# -----------------------------------------------------------------------------
# Abstract base class declaring interface methods
# -----------------------------------------------------------------------------

class Rewards(ABC):
    """
    Abstract base class for concrete reward classes.
    
    The class declares the methods that must be callable from the labyrinth
    environments. If subclasses are implemented based on these methods, they
    can easily be integrated and exchanged in the environment.
    
    """

    # ========== Abstract methods =============================================

    @abstractmethod
    def __init__(self, layout):
        """
        Constructor.

        Parameters
        ----------
        layout : enum Layout
            Layout of holes and walls as defined in LabLayouts.py.

        Returns
        -------
        None.

        """
        pass
    
    # -------------------------------------------------------------------------

    @abstractmethod
    def step(self, state, action, next_state, is_near_hole, is_in_hole, is_at_destination):
        """
        Get reward for a action applied in the environment.

        Parameters
        ----------
        state : numpy.array
            Observation as defined in LabyrinthEnv before the action.
        action : numpy.array
            Action performed as defined in LabyrinthEnv.
        next_state : numpy.array
            Observation as defined in LabyrinthEnv after the action.
        is_near_hole : bool
            True if the ball is near a hole, else False.
        is_in_hole : bool
            True if the ball is in a hole, else False.
        is_at_destination : bool
            True if the ball is at the destination, else False.

        Returns
        -------
        int
            Reward for step

        """
        pass

    # ========== Reset ========================================================

    def reset(self):
        """
        Reset.
        
        There are no attributes to reset in this base class. May be overriden
        by sub classes.

        Returns
        -------
        None.

        """
        pass
    
# -----------------------------------------------------------------------------
# Reward class based on passing areas (regions, tiles) of the labyrinth
# -----------------------------------------------------------------------------

class RewardsByAreas(Rewards):
    """
    Reward system based on regions.

    The labyrinth's boards get segmented in rectangular areas (= regions, tiles).
    For each area, there is a defined target point where the ball should move to
    in order to move into a neighboring area bringing the ball to the labyrinth's
    destination.
    
    """
    
    # TODO Why not reward distance to center for layouts with 0 holes? Segments are purely cosmetic for visual assistance.

    # ========== Constructor ==================================================

    def __init__(self, layout):
        """
        Contructor.
        
        See documentation in base class Rewards.
        
        """
        # Layout
        self.__layout = layout
        
        # Area coordinates and destination locations
        self.areas = self.__areas(layout)
        self.target_points = self.__target_points(layout)
        
        # Reward values
        self.__init_reward_values(layout)

    # ========== Determine reward =============================================

    def step(self, state, action, next_state, is_near_hole, is_in_hole, is_at_destination):
        """
        Get reward for a action applied in the environment.

        See documentation in base class Rewards.

        """
        # Distance to target point before/after taking action
        last_distance, next_distance = self.__distances_from_target_point(state=state, next_state=next_state)
        delta_distance = last_distance - next_distance
        
        # Determine reward
        if self.__layout.number_holes > 0:
            # Layouts with holes
            if is_at_destination:
                reward = self.__destination
                print("Ball reached destination")
            elif is_in_hole:
                reward = self.__in_hole
                print("Ball lost")
            elif is_near_hole:
                reward = self.__near_hole
                print("Ball near hole")
            elif delta_distance > 0.25:
                reward = self.__interim(self.__progress, self.areas)
            elif delta_distance > 0.0:
                reward = self.__correct_direction
            else:
                reward = self.__default

        else:
            # Layouts with 0 holes
            progress = self.__reward_for_layout_0_holes(next_distance)
            if is_at_destination:
                print("Ball reached destination")
                reward = self.__destination
            elif delta_distance > 0.0:
                reward = self.__interim_dict.get(progress, self.__default)
                if reward == 100:
                    print("Ball near center")
            else:
                reward = self.__default_dict.get(progress, self.__default)
                
        return reward

    # ----------- Ball moves in the correct direction -------------------------
    
    def __distances_from_target_point(self, state, next_state):
        """
        Calculate distances to target point before and after taking the action.

        Parameters
        ----------
        state : numpy.array
            Observation as defined in LabyrinthEnv before the action.
        next_state : numpy.array
            Observation as defined in LabyrinthEnv after the action.

        Returns
        -------
        last_distance : float
            Distance before taking the action.
        next_distance : float
            Distance after taking the action.

        """
        # Init return values
        last_distance = 0.0
        next_distance = 0.0

        # Ball positions
        last_x = state[0]
        last_y = state[1]
        next_x = next_state[0]
        next_y = next_state[1]
        
        # Progress (in number areas) of movement through labyrinth. Value increases from the goal to start, because the areas are defined from the goal to the start.
        self.__progress = 0
        
        # Run through defined areas
        for x_min, x_max, y_min, y_max in self.areas:
            is_in_area_x = (x_min < last_x) and (x_max > last_x)
            is_in_area_y = (y_min < last_y) and (y_max > last_y)

            # Found area the ball was in during last step            
            if is_in_area_x and is_in_area_y:
                target_point_x = self.target_points[self.__progress][0]
                target_point_y = self.target_points[self.__progress][1]
           
                # TODO Replace positions and points by single object (e.g., [self.__ball_position.x, self.__ball_position.y] -> self.__ball_position)?
                last_distance = math.dist([last_x, last_y], [target_point_x, target_point_y])
                next_distance = math.dist([next_x, next_y], [target_point_x, target_point_y])               
                break
                
            self.__progress += 1

        return last_distance, next_distance
            
    # ----------- Layouts without hole: Distance to center --------------------
    
    def __reward_for_layout_0_holes(self, distance):
        """
        Calculates in witch circular segment the ball is located for the layouts
        Layout.HOLES_0_VIRTUAL and Layout.HOLES_0.

        Parameters
        ----------
        distance : float
            Distance to center after taking the action.

        Returns
        -------
        radius_progress: int
            The higher the number, the closer the ball is to the center
        """
        radii = [10, 7.5, 5, 2.5, 1.25]
        radius_progress = 1

        for radius in radii:
            if distance > radius:
                return radius_progress
            radius_progress += 1
            
        return radius_progress

    # ========== Defined areas and target points ==============================
    
    def __areas(self, layout):
        """
        Segments the labyrinth's board into rectangular areas.
        
        Each entry represents a rectangular region by [x_min, x_max, y_min, y_max].
        For each layout, the list of areas is sorted from the goal/destination
        to the starting location.

        Parameters
        ----------
        layout : enum Layout
            Layout of holes and walls as defined in LabLayouts.py.

        Returns
        -------
        Array of float[4]
            Boundaries of areas from destination to starting location.

        """
        areas = {
            Layout.HOLES_0_VIRTUAL  : [[-13.06, 13.06, -10.76, 10.76]],
            Layout.HOLES_0          : [[-13.06, 13.06, -10.76, 10.76]],
            Layout.HOLES_2_VIRTUAL : [
                [-3.15, 3.3, -6.55, -4.38], [-3.15, 3.3, -4.38, -1.02], [-3.15, 1.02, -1.02, 4.16], [1.02, 3.3, 0.5, 4.16], [-0.74, 3.3, 4.16, 6.44],
                [-0.74, 3.3, 6.44, 11.4],   [-3.15, -0.74, 6.44, 11.4]],
            Layout.HOLES_2 : [
                [-2.09, 4.6, -9.9, -6.05], [-2.09, 1.96, -6.05, -2.92], [-2.09, 1.96, -2.92, 2.28], [1.96, 4.6, -2.92, 2.28], [0.14, 4.6, 2.28, 5.03],
                [0.14, 4.6, 5.03, 10.27],  [-2.09, 0.14, 5.03, 10.27]],
            Layout.HOLES_8 : [
                [-5.86, 0.14, -11.40, -9.52],  [0.14, 7.46, -11.40, -9.52],   [5.21, 7.46, -9.52, -4.63],   [7.46, 9.46, -11.4, -4.63],  [9.46, 13.7, -11.4, -6.58],
                [9.46, 13.7, -6.58, -3.98],    [9.46, 13.7, -3.98, -0.65],    [9.46, 13.7, -0.65, 1.9],     [9.46, 13.7, 1.9, 4.15],     [9.46, 13.7, 4.15, 5.75],
                [9.46, 13.7, 5.75, 7.28],      [8.23, 13.7, 7.28, 11.4],      [2.26, 8.23, 9.09, 11.4],     [2.26, 7.83, 7.08, 9.09],    [4.81, 8.82, 2.16, 7.08],
                [-1.98, 4.81, 0.6, 4.11],      [-1.98, 0.29, -1.23, 0.6],     [-1.98, 0.29, -3.56, -1.23],  [0.29, 4.59, -3.85, -1.23],  [0.16, 4.59, -6.95, -3.85],
                [-3.38, 0.16, -8.86, -4.17],   [-6.26, -3.38, -8.86, -4.17],  [-8.88, -6.26, -6.8, -4.17],  [-8.88, -6.26, -11.4, -6.8], [-13.7, -8.88, -11.4, -6.53],
                [-13.7, -11.59, -6.53, -4.11], [-13.7, -10.25, -4.11, -2.64], [-12.1, -8.98, -2.64, -0.16], [-8.98, -4.73, -3.53, 0.11], [-4.73, -2.53, -3.53, 2.01],
                [-6.58, -2.53, 2.01, 5.56],    [-10.13, -6.58, 2.39, 5.56],   [-11.69, -10.13, 0.61, 5.56], [-13.7, -11.69, 0.61, 4.68], [-13.7, -11.66, 4.68, 11.4],
                [-11.66, -5.34, 9.99, 11.4],   [-5.34, 0.5, 9.99, 11.4]]
        }
        return areas[layout]
    
    # -------------------------------------------------------------------------
    
    def __target_points(self, layout):
        """
        Define target points where to move the ball to in each area.
        
        An array is returned with the same number of entries as the areas
        returned by self.__areas(). Each entry contains the local destination
        coordinate [x, y] to move to in the corresponing area.
        
        An agend can reward movements into the direction of the 'target point'
        of the area the ball is currently in.

        Parameters
        ----------
        layout : enum Layout
            Layout of holes and walls as defined in LabLayouts.py.

        Returns
        -------
        Array of float[2]
            Destination location of areas from overall destination to starting location.

        """
        target_points = {
            Layout.HOLES_0_VIRTUAL  : [[0, 0]],
            Layout.HOLES_0          : [[0, 0]],
            Layout.HOLES_2_VIRTUAL : [
                [-0.7, -5.98], [-0.7, -5.98], [-0.7, -5.98], [0.17, 1.63], [0.17, 1.63],
                [1.51, 5.72],  [0.31, 7.92]],
            Layout.HOLES_2 : [
                [1.24, -10.42], [0.3, -6.77], [-0.22, -3.57], [1.06, -0.92], [1.06, -0.92],
                [2.71, 4.28],   [1.13, 6.55]],
            Layout.HOLES_8 : [
                [-4.4, -10.32],  [-4.4, -10.32], [6.92, -10.32],  [7.33, -6.88],  [8.89, -8.29],
                [12.54, -7.53],  [12.99, -5],    [10.35, -1.15],  [12.53, -1.51], [12.41, 3.74],
                [11.39, 5.36],   [11.43, 7.01],  [8.71, 10.19],   [4.54, 9.6],    [6.07, 7.55],
                [5.52, 3.39],    [-1.2, 1.14],   [-1.22, -0.73],  [-0.05, -2.6],  [3.35, -2.99],
                [0.94, -5.23],   [-2.74, -7.98], [-5.79, -5.36],  [-7.59, -6.12], [-8.1, -7.6],
                [-12.58, -7.03], [-13.16, -4.7], [-11.63, -3.18], [-9.8, -1.49],  [-5.55, -1.84],
                [-3.31, 1.29],   [-6.09, 4.15],  [-9.51, 4.48],   [-11.05, 3.58], [-12.85, 3.92],
                [-12.45, 10.83], [-12.45, 10.83]]
        }
        return target_points[layout]

        
    # ========== Define rewards ===============================================
            
    def __init_reward_values(self, layout):
        """
        Define the rewards to be granted for taking steps in an environment.
        
        Note that the attributes initialized with reward values differ for
        layouts with and without holes.

        Parameters
        ----------
        layout : enum Layout
            Layout of holes and walls as defined in LabLayouts.py.

        Returns
        -------
        None

        """
        match layout:
            case Layout.HOLES_2_VIRTUAL | Layout.HOLES_2:
                self.__destination = 600
                self.__in_hole = -200
                self.__near_hole = -10
                self.__correct_direction = -1
                self.__interim = lambda progress, areas: 3 / len(areas) * (len(areas) - progress)
                self.__default = -2
                
            case Layout.HOLES_8:
                self.__destination = 1000
                self.__in_hole = -300
                self.__near_hole = -15
                self.__correct_direction = -1
                self.__interim = lambda progress, areas: 6 / len(areas) * (len(areas) - progress)
                self.__default = -2

            # The higher the key for the dictionaries, the closer to the center the circle element in which the ball is located
            case Layout.HOLES_0_VIRTUAL | Layout.HOLES_0:
                self.__destination = 600
                self.__default = -2
                self.__interim_dict = {
                    6: 100,
                    5: 25,
                    4: 2,
                    3: -0.2,
                    2: -0.4 }
                self.__default_dict = {
                    6: -0.2,
                    5: -0.4 }
                
            case _:
                raise Exception(f'Layout not supported: {layout}')
