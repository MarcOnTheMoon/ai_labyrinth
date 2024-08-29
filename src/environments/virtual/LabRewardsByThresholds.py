"""
Reward system based on passing gates/thresholds for a labyrinth OpenAI gym environment.

This file is work-in-progress and NOT ready for use, yet.

@authors: Sandra Lassahn, Marc Hensel
@contact: http://www.haw-hamburg.de/marc-hensel
@copyright: 2024
@version: 2024.08.29
@license: CC BY-NC-SA 4.0, see https://creativecommons.org/licenses/by-nc-sa/4.0/deed.en
"""
from LabLayouts import Layout

class RewardsByThresholds():

    # ========== Constructor ==================================================

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
        # TODO Reward dictionary
        # self.reward_dict = self.__reward_dictionary(layout)
        
        # Thresholds
        self.thresholds = self.__thresh_reward_dictionary(layout)
        
        # Counter
        # The "reward counter" is confusing. What is this? A counter? An index to an array? ...? -> Answer: Not a counter, but index (number thresholds successfully passed)
        # TODO Set to 0 on environment.reset()
        self.__is_moved_in_correct_direction_counter = 0   # Interim reward counter
        
    # ========== Defined thresholds ===========================================
   
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
        layout : enum Layout
            Layout of holes and walls as defined in LabLayouts.py.

        Returns
        -------
        threshold_rewards : array of [char, int, float, float, float]
            Array of thresholds to pass from start to destination.

        """
        match layout:
            case Layout.HOLES_2_VIRTUAL:
                threshold_rewards = [
                    ['x', 1, -0.93, 4.21, 11.4], ['y', -1, 4.21, -1.04, 3.28], ['y', -1, -0.96, -3.15, 3.28], ['y', -1, -4.35, -1.81, 1.71]]
            
            case Layout.HOLES_2:
                threshold_rewards = [
                    ['x', 1, 0.14, 2.52, 11.4], ['y', -1, 2.25, 0.14, 4.57], ['y', -1, -5.26, -2.13, 4.57], ['y', -1, -8.8, -0.27, 2.81]]
            
            case Layout.HOLES_8:
                threshold_rewards = [
                    ['x', -1, -4.63, 9.99, 11.4],    ['x', -1, -10.07, 9.99, 11.4], ['y', -1, 7.09, -13.7, -11.87], ['x', 1, -11.66, 0.45, 4.38],  ['x', 1, -10.10, 2.13, 5.64],
                    ['x', 1, -6.57, 2.13, 5.64],     ['y', -1, 2.09, -6.21, -2.52], ['x', -1, -4.94, -3.56, -0.2],  ['x', -1, -8.98, -2.76, -0.2], ['y', -1, -2.79, -12.01, -9.18],
                    ['y', -1, -6.31, -13.7, -10.95], ['x', 1, -10.95, -11.4, -6.8], ['x', 1, -8.91, -11.4, -6.8],   ['y', 1, -6.53, -8.64, -6.37], ['x', 1, -6.19, -6.65, -4.24],
                    ['x', 1, -3.27, -8.86, -6.69],   ['x', 1, 0.19, -7.45, -4.28],  ['y', 1, -3.75, 2.03, 4.61],    ['x', -1, 1.28, -3.48, -0.89], ['y', 1, -0.16, -1.95, -2.66],
                    ['x', 1, 1.45, 0.18, 3.77],      ['y', 1, 4.15, 3.42, 8.78],    ['y', 1, 7.13, 3.78, 7.66],     ['y', 1, 9.34, 2.26, 6.1],     ['x', 1, 7.97, 9.6, 11.4],
                    ['y', -1, 7.43, 10.62, 13.7],    ['y', -1, 4.15, 10.82, 13.7],  ['y', -1, -0.58, 9.36, 11.74],  ['y', -1, -4.13, 11.03, 13.7], ['y', -1, -6.34, 10.52, 13.7],
                    ['x', -1, 10.52, -11.4, -6.34],  ['x', -1, 7.8, -8.17, -4.66],  ['y', -1, -9.13, 5.18, 7.46],   ['x', -1, 0.53, -11.4, -9.55]]
                
            case _:
                threshold_rewards = {}
                
        return threshold_rewards

    # ----------- Threshold (currently not used) ------------------------------
    
    # TODO Method has been copied from environment and needs to be fixed (e. g., attributes like ball position undefined in this class)
    
    def __threshold_reward(self):
        """
        Calculates whether a defined threshold is crossed.
        
        The method is not used so far, but can be used to try thresholding rewards.

        Parameters
        ----------
        None

        Returns
        -------
        1, if the threshold was crossed in the correct direction.
        -1, if the threshold was crossed in the wrong direction.
        0, otherwise

        """
        # TODO Code review
        if len(self.__reward_area.threshold_rewards) <= self.__is_moved_in_correct_direction_counter:
            return 0

        axis = self.__reward_area.threshold_rewards[self.__is_moved_in_correct_direction_counter][0]
        direction = self.__reward_area.threshold_rewards[self.__is_moved_in_correct_direction_counter][1]
        threshold = self.__reward_area.threshold_rewards[self.__is_moved_in_correct_direction_counter][2]
        range_min = self.__reward_area.threshold_rewards[self.__is_moved_in_correct_direction_counter][3]
        range_max = self.__reward_area.threshold_rewards[self.__is_moved_in_correct_direction_counter][4]
        if axis == 'x':
            if direction < 0:
                if self.__last_ball_position.x > threshold and self.__ball_position.x <= threshold and self.__ball_position.y > range_min and self.__ball_position.y < range_max:
                    self.__is_moved_in_correct_direction_counter += 1  # right crossing direction
                    return 1
                elif self.__last_ball_position.x < threshold and self.__ball_position.x >= threshold and self.__ball_position.y > range_min and self.__ball_position.y < range_max:
                    self.__is_moved_in_correct_direction_counter -= 1  # false crossing direction
                    return -1
            elif self.__last_ball_position.x < threshold and self.__ball_position.x >= threshold and self.__ball_position.y > range_min and self.__ball_position.y < range_max:
                self.__is_moved_in_correct_direction_counter += 1  # right crossing direction
                return 1
            elif self.__last_ball_position.x > threshold and self.__ball_position.x <= threshold and self.__ball_position.y > range_min and self.__ball_position.y < range_max:
                self.__is_moved_in_correct_direction_counter -= 1  # false crossing direction
                return -1
        else:  # axis == 'y'
            if direction < 0:
                if self.__last_ball_position.y > threshold and self.__ball_position.y <= threshold and self.__ball_position.x > range_min and self.__ball_position.x < range_max:
                    self.__is_moved_in_correct_direction_counter += 1  # right crossing direction
                    return 1
                elif self.__last_ball_position.y < threshold and self.__ball_position.y >= threshold and self.__ball_position.x > range_min and self.__ball_position.x < range_max:
                    self.__is_moved_in_correct_direction_counter -= 1  # false crossing direction
                    return -1
            elif self.__last_ball_position.y < threshold and self.__ball_position.y >= threshold and self.__ball_position.x > range_min and self.__ball_position.x < range_max:
                self.__is_moved_in_correct_direction_counter += 1 # right crossing direction
                return 1
            elif self.__last_ball_position.y > threshold and self.__ball_position.y <= threshold and self.__ball_position.x > range_min and self.__ball_position.x < range_max:
                self.__is_moved_in_correct_direction_counter -= 1 # false crossing direction
                return -1
        return 0
