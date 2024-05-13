# https://www.gymlibrary.dev/content/environment_creation/

import gym
from gym import spaces
import numpy as np
from math import pi

from vpython import vector as vec
from LabyrinthRender3D import LabyrinthRender3D
from LabyrinthGeometry import LabyrinthGeometry
from LabyrinthBall import LabyrinthBall

import time


# -----------------------------------------------------------------------------
# Environment
# -----------------------------------------------------------------------------

class LabyrinthEnvironment(gym.Env):
    metadata = {'render_modes': '3D'} #render mode

    def __init__(self, render_mode='3D'):

        #action space
        self.action_space = spaces.Discrete(9) # init possible actions
        self.__action_to_angle = {
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

        assert render_mode is None or render_mode in self.metadata["render_modes"] #stellt sicher das render mode entweder none ist oder in der liste metadata
        self.render_mode = render_mode

        #initial render
        if self.render_mode == '3D':
            geometry = LabyrinthGeometry()
            # self.__ball_start_position = vec(-1.52, 9.25, 0) #field with 2 holes
            self.__ball_start_position = vec(0.13, 10.53, 0)  # field with 8 holes
            self.render_window = LabyrinthRender3D(geometry, ball_position=self.__ball_start_position)
            self.calcBall = LabyrinthBall(geometry)
            self.__field_rotation_x = 0
            self.__field_rotation_y = 0
            self.__ball_position = self.calcBall.calc_move(self.__field_rotation_x, self.__field_rotation_y, self.__ball_start_position)

        #observation space
        self.observation_space = spaces.Dict({
            'ball_position': spaces.Box(low=-np.inf, high=np.inf, shape=(2,), dtype=np.float32),
            'ball_velocity': spaces.Box(low=-np.inf, high=np.inf, shape=(2,), dtype=np.float32),
            'field_rotation': spaces.Box(low=-np.inf, high=np.inf, shape=(2,), dtype=np.float32)
        })

        self.observation_space, _ = self.reset() #initial zustand, zweiter parameter leer info aber erforderlich für gym

    # ======================= Reset environment ==========================

    def reset(self):
        self.last_action = 0
        self.number_actions = 0

        self.observation_space = {
            'ball_position': np.array([self.__ball_start_position.x, self.__ball_start_position.y]),
            'ball_velocity': np.array([0.0, 0.0]),
            'field_rotation': np.array([0.0, 0.0])
        }

        #reset render to start position
        self.render_window.rotate_to(0,0)
        self.render_window.move_ball(self.__ball_start_position.x,self.__ball_start_position.y)
        self.render_window.ball_visibility(True)
        self.calcBall.calc_move(0, 0, self.__ball_start_position)

        return self.observation_space, {}

    # ======================= step =================================
    def step(self, action):

        # update observation_space
        self.__field_rotation_x = float(self.__action_to_angle[action][0])
        self.__field_rotation_y = float(self.__action_to_angle[action][1])
        self.observation_space = {
            'ball_position': np.array([self.__ball_position.x, self.__ball_position.y]),
            'ball_velocity': np.array([self.calcBall.get_velocity().x, self.calcBall.get_velocity().y]),
            'field_rotation': np.array([self.__field_rotation_x, self.__field_rotation_y])
        }
        self.__done = False
        # is destination reached?
        self.__terminated = False
        self.__destination_x = [-5.85, -3.83]  # field with 8 holes
        self.__destination_y = [-11.4, -9.52]  # field with 8 holes
        if self.__ball_position.x > self.__destination_x[0] and self.__ball_position.x < self.__destination_x[1] and self.__ball_position.y > self.__destination_y[0] and self.__ball_position.y < self.__destination_y[1]:
            self.__terminated = True
            self.__done = True

        # is the game lost (ball fall in a hole)?
        self.__lost = False
        if self.calcBall.game_state == -1:
            self.__lost = True
            self.__done = True


        # reward
        if self.__terminated:
            print("Terminated")
            reward = 500000
        elif self.__lost:
            print("lost")
            reward = -1000000
        else:
            reward = -1


        self.last_action = action
        self.number_actions += 1

        return self.observation_space, reward, self.__done, {}

    # ======================= render environment =================================
    def render(self, action):
        if self.render_mode in ['3D']:
            self.render_window.rotate_to(float(self.__action_to_angle[action][0]), float(self.__action_to_angle[action][1]))

    #erst rendern dann step

    def render_ball(self, action):
        #continuous calculation of the ball position
        self.__x_rad = float(self.__action_to_angle[action][0]) * pi/180.0
        self.__y_rad = float(self.__action_to_angle[action][1]) * pi/180.0
        self.__ball_position = self.calcBall.calc_move(self.__x_rad, self.__y_rad)
        #print(self.__ball_position)
        if self.calcBall.game_state == -1:
            self.render_window.ball_visibility(False)
        self.render_window.move_ball(self.__ball_position.x, self.__ball_position.y, x_rad=self.__x_rad, y_rad=self.__y_rad,)
        time.sleep(0.01)


if __name__ == '__main__':
    for render_mode in ['3D']:
        env = LabyrinthEnvironment(render_mode=render_mode)

        for action in [2,6,3,1,0,5,5,5,5,5,5,5,5]:
            env.render(action)
            for i in range(8):
                env.render_ball(action)
            env.step(action)