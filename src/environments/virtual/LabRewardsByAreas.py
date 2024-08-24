"""
Reward system based on regions for a labyrinth OpenAI gym environment.

The labyrinth's boards get segmented in rectangular areas (= regions, tiles).
For each area, there is a defined target point where the ball should move to
in order to move into a neighboring area bringing the ball to the labyrinth's
destination.

@authors: Sandra Lassahn, Marc Hensel
@contact: http://www.haw-hamburg.de/marc-hensel
@copyright: 2024
@version: 2024.08.24
@license: CC BY-NC-SA 4.0, see https://creativecommons.org/licenses/by-nc-sa/4.0/deed.en
"""
from LabLayouts import Layout

class RewardsByAreas():
    
    # TODO Why not reward distance to center for layouts with 0 holes? Segments are purely cosmetic for visual assistance.

    # ========== Constructor ==================================================

    def __init__(self, layout):
        """
        Constructor.

        Parameters
        ----------
        layout: String
            Layout of holes and walls as defined in LabGeometry.py.

        Returns
        -------
        None.

        """
        # Reward dictionary
        self.reward_dict = self.__reward_dictionary(layout)
        
        # Area coordinates and destination locations
        self.areas = self.__areas(layout)
        self.target_points = self.__target_points(layout)
        
    # ========== Defined areas and target points ==============================
    
    def __areas(self, layout):
        """
        Segments the labyrinth's board into rectangular areas.
        
        Each entry represents a rectangular region by [x_min, x_max, y_min, y_max].
        For each layout, the list of areas is sorted from the goal/destination
        to the starting location.

        Parameters
        ----------
        layout : enum LabLayouts.Layout
            Layout of holes and walls as defined in LabGeometry.py.

        Returns
        -------
        Array of float[4]
            Boundaries of areas from destination to starting location.

        """
        areas = {
            Layout.HOLES_0 :      [[-13.06, 13.06, -10.76, 10.76]],
            Layout.HOLES_0_REAL : [[-13.06, 13.06, -10.76, 10.76]],
            Layout.HOLES_2 : [
                [-3.15, 3.3, -6.55, -4.38], [-3.15, 3.3, -4.38, -1.02], [-3.15, 1.02, -1.02, 4.16], [1.02, 3.3, 0.5, 4.16], [-0.74, 3.3, 4.16, 6.44],
                [-0.74, 3.3, 6.44, 11.4],   [-3.15, -0.74, 6.44, 11.4]],
            Layout.HOLES_2_REAL : [
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
        layout : enum LabLayouts.Layout
            Layout of holes and walls as defined in LabGeometry.py.

        Returns
        -------
        Array of float[2]
            Destination location of areas from overall destination to starting location.

        """
        target_points = {
            Layout.HOLES_0 :      [[0, 0]],
            Layout.HOLES_0_REAL : [[0, 0]],
            Layout.HOLES_2 : [
                [-0.7, -5.98], [-0.7, -5.98], [-0.7, -5.98], [0.17, 1.63], [0.17, 1.63],
                [1.51, 5.72],  [0.31, 7.92]],
            Layout.HOLES_2_REAL : [
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
            
    def __reward_dictionary(self, layout):
        """
        Define the rewards to be granted for taking steps in an environment.

        Parameters
        ----------
        layout : enum LabLayouts.Layout
            Layout of holes and walls as defined in LabGeometry.py.

        Returns
        -------
        reward_dict : dictionary
            Dictionary with rewards for defined observations

        """
        if layout.number_holes == 2:
            reward_dict =  {
                'is_ball_at_destination': 600,
                'is_ball_in_hole': -200,
                'is_ball_to_close_hole': -10,
                'interim_reward': lambda progress, areas: 3 / len(areas) * (len(areas) - progress),
                'right_direction': -1,
                'default': -2}
            
        elif layout == Layout.HOLES_8:
            reward_dict =  {
                'is_ball_at_destination': 1000,
                'is_ball_in_hole': -300,
                'is_ball_to_close_hole': -15,
                'interim_reward': lambda progress, areas: 6 / len(areas) * (len(areas) - progress),
                'right_direction': -1,
                'default': -2}

        #The higher the first number for interim or default, the closer the circle element is to the center (smaller the circle) in which the ball is located
        elif layout.number_holes == 0:
            reward_dict =  {
                'destination': 600,
                'interim': {
                    6: 100,
                    5: 25,
                    4: 2,
                    3: -0.2,
                    2: -0.4,
                    'default': -2},
                'default': {
                    6: -0.2,
                    5: -0.4,
                    'default': -2 }}
        else:
            reward_dict = {}
            
        return reward_dict
    
    # -------------------------------------------------------------------------

    # TODO Not related to areas => Move into own reward class
    def __thresh_reward_dictionary(self, layout):
        """
        Define threshold rewards (currently NOT used).

        The idea is to positively reward passing defined crossings. The elements
        contain values [coordinate x or y of crossing, direction of crossing, threshold, min range, max range]
        with direction of crossing being
        
            +1 = from smaller to larger values,
            -1 = from larger to smaller values.
        
        The order of thresholds is defined from the start to the goal.

        Parameters
        ----------
        layout : enum LabLayouts.Layout
            Layout of holes and walls as defined in LabGeometry.py.

        Returns
        -------
        threshold_rewards : array of [char, int, float, float, float]
            Array of thresholds to pass from start to destination.

        """
        if self.__layout == Layout.HOLES_2:
            threshold_rewards = [
                ['x', 1, -0.93, 4.21, 11.4], ['y', -1, 4.21, -1.04, 3.28], ['y', -1, -0.96, -3.15, 3.28], ['y', -1, -4.35, -1.81, 1.71]]
            
        elif self.__layout == Layout.HOLES_2_REAL:
            threshold_rewards = [
                ['x', 1, 0.14, 2.52, 11.4], ['y', -1, 2.25, 0.14, 4.57], ['y', -1, -5.26, -2.13, 4.57], ['y', -1, -8.8, -0.27, 2.81]]
            
        elif self.__layout == Layout.HOLES_8:
            threshold_rewards = [
                ['x', -1, -4.63, 9.99, 11.4],    ['x', -1, -10.07, 9.99, 11.4], ['y', -1, 7.09, -13.7, -11.87], ['x', 1, -11.66, 0.45, 4.38],  ['x', 1, -10.10, 2.13, 5.64],
                ['x', 1, -6.57, 2.13, 5.64],     ['y', -1, 2.09, -6.21, -2.52], ['x', -1, -4.94, -3.56, -0.2],  ['x', -1, -8.98, -2.76, -0.2], ['y', -1, -2.79, -12.01, -9.18],
                ['y', -1, -6.31, -13.7, -10.95], ['x', 1, -10.95, -11.4, -6.8], ['x', 1, -8.91, -11.4, -6.8],   ['y', 1, -6.53, -8.64, -6.37], ['x', 1, -6.19, -6.65, -4.24],
                ['x', 1, -3.27, -8.86, -6.69],   ['x', 1, 0.19, -7.45, -4.28],  ['y', 1, -3.75, 2.03, 4.61],    ['x', -1, 1.28, -3.48, -0.89], ['y', 1, -0.16, -1.95, -2.66],
                ['x', 1, 1.45, 0.18, 3.77],      ['y', 1, 4.15, 3.42, 8.78],    ['y', 1, 7.13, 3.78, 7.66],     ['y', 1, 9.34, 2.26, 6.1],     ['x', 1, 7.97, 9.6, 11.4],
                ['y', -1, 7.43, 10.62, 13.7],    ['y', -1, 4.15, 10.82, 13.7],  ['y', -1, -0.58, 9.36, 11.74],  ['y', -1, -4.13, 11.03, 13.7], ['y', -1, -6.34, 10.52, 13.7],
                ['x', -1, 10.52, -11.4, -6.34],  ['x', -1, 7.8, -8.17, -4.66],  ['y', -1, -9.13, 5.18, 7.46],   ['x', -1, 0.53, -11.4, -9.55]]
        else:
            threshold_rewards = {}
                
        return threshold_rewards
