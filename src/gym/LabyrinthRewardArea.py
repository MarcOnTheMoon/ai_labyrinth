
class LabyrinthRewardArea():

    def __init__(self, layout):
        self.layout = layout

        # Für Fortschritt
        # Reihnfolge ziel nach anfang, mit jeweils [x,y] Zwischenzielkoordinaten
        if self.layout == '0 holes':
            self.target_points = [[0, 0]]
    
        elif self.layout == '2 holes':
            self.target_points = [[-0.13, -6],
                             [-0.13, -1.52],
                             [0.33, 1.2],
                             [0.33, 1.2],
                             [1.97, 5.81]]
    
        elif self.layout == '2 holes real':
            self.target_points = [[1.24, -10.42],
                             [0.3, -6.77],
                             [-0.22, -3.57],
                             [1.06, -0.92],
                             [1.06, -0.92],
    
                             [2.71, 4.28],
                             [1.13, 6.55]]
    
        elif self.layout == '8 holes':
            self.target_points = [[-4.4, -10.32],
                             [-4.4, -10.32],
                             [6.92, -10.32],
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
                             [-7.56, 6.12],
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
        if self.layout == '0 holes':
            self.areas = [[-13.06, 13.06, -10.76, 10.76]]
    
        elif self.layout == '2 holes':
            self.areas = [[-3.13, 3.33, -6.53, -1.03],
                          [-3.13, 1.0, -1.03, 4.08],
                          [1.0, 3.33, 0.47, 4.08],
                          [-0.7, 3.33, 4.08, 6.48],
                          [-3.13, 3.33, 6.48, 11.4]]
    
        elif self.layout == '2 holes real':
            self.areas = [[-2.09, 4.6, -9.9, -6.05],
                          [-2.09, 1.96, -6.05, -2.92],
                          [-2.09, 1.96, -2.92, 2.28],
                          [1.96, 4.6, -2.92, 2.28],
                          [0.14, 4.6, 2.28, 5.03],
    
                          [0.14, 4.6, 5.03, 10.27],
                          [-2.09, 0.14, 5.03, 10.27]]
    
        elif self.layout == '8 holes':
            self.areas = [[-5.86, 0.14, -11.40, -9.52],
                          [0.14, 7.46, -11.40, -9.52],
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
                          [-8.88, -6.26, -6.8, -4.17],
                          [-8.88, -6.26, -11.4, -6.8],
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
