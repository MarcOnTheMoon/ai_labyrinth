"""
The pulse width is determined from the desired angular positions according to
the used servos and transmitted to the Arduino via a serial interface.

@authors: Sandra Lassahn
@contact: http://www.haw-hamburg.de/marc-hensel
@copyright: 2024
@version: 2024.04.25
@license: CC BY-NC-SA 4.0, see https://creativecommons.org/licenses/by-nc-sa/4.0/deed.en
"""

import math
import time
from ArduinoCOM import ArduinoCOM

class ServoCommunication:

    # ========== Constructor ==================================================
    def __init__(self, port=None):
        """
        Constructor.

        Parameters
        ----------
        port : String, optional
            Port number for serial communication to the arduino

        Returns
        -------
        None.
        """
        self.__arduino = ArduinoCOM(serialCOM = port, baudRate = 115200, readTimeoutSec = .1)

        self.__channel = [1, 0] #Connected channel of the servo to the PWM driver, [x,y]
        self.__pulse_width_max = [2650, 2540] #The maximum and minimum pulse width is different for each servo
        self.__pulse_width_min = [350, 420]
        self.__pulse_middle = [((self.__pulse_width_max[0] + self.__pulse_width_min[0]) / 2), 
                               ((self.__pulse_width_max[1] + self.__pulse_width_min[1]) / 2)]
        self.__pulse_per_degree = [((self.__pulse_width_max[0] - self.__pulse_middle[0])/(math.asin(8.6/155.5)*180/math.pi)), 
                                   ((self.__pulse_width_max[1] - self.__pulse_middle[1]) / (math.asin(5.2 / 122.5) * 180 / math.pi))]
        self.__degree = [0.0, 0.0]
        self.__min_degree = [((self.__pulse_width_min[0] - self.__pulse_middle[0]) / self.__pulse_per_degree[0]), 
                             ((self.__pulse_width_min[1] - self.__pulse_middle[1]) / self.__pulse_per_degree[1])]
        self.__max_degree = [((self.__pulse_width_max[0] - self.__pulse_middle[0]) / self.__pulse_per_degree[0]),
                             ((self.__pulse_width_max[1] - self.__pulse_middle[1]) / self.__pulse_per_degree[1])]

    # ========== transmit message to the Arduino =========================================
    def __write_pulse(self, channel):
        """
        Combines all data into a string for the message and transmits it to the Arduino via the serial interface.
        sent data: channel and pulse width for one servo

        Parameters
        ----------
        channel: int

        Returns
        -------
        String

        """
        if channel == self.__channel[0]:
            command = str(channel) + ";" + str(self.__x_pulse)
        elif channel == self.__channel[1]:
            command = str(channel) + ";" + str(self.__y_pulse)

        print(command)
        self.__arduino.writeData(command)
        time.sleep(0.05)
        data = self.__arduino.readLine()
        return data

    # ========== Calculate degree to pulse ====================================
    def __calc_x_degree_to_pulse(self):
        """
        Calculates the corresponding pulse length for the servo that
        tilts the playing field in the x direction from the desired angle

        Parameters
        ----------
        None.

        Returns
        -------
        None.
        """
        self.__x_pulse = int(self.__degree[0] * self.__pulse_per_degree[0] + self.__pulse_middle[0])
        #Check whether the calculated pulse width is within the permitted range, otherwise limit it
        if self.__x_pulse < self.__pulse_width_min[0]:
            self.__x_pulse = self.__pulse_width_min[0]
            self.__degree[0] = self.__min_degree[0]
        elif self.__x_pulse > self.__pulse_width_max[0]:
            self.__x_pulse = self.__pulse_width_max[0]
            self.__degree[0] = self.__max_degree[0]

    # -------------------------------------------------------------------------
    def __calc_y_degree_to_pulse(self):
        """
        Calculates the corresponding pulse length for the servo that
        tilts the playing field in the y direction from the desired angle

        Parameters
        ----------
        None.

        Returns
        -------
        None.
        """
        self.__y_pulse = int(self.__degree[1] * self.__pulse_per_degree[1] + self.__pulse_middle[1])
        # Check whether the calculated pulse width is within the permitted range, otherwise limit it
        if self.__y_pulse < self.__pulse_width_min[1]:
            self.__y_pulse = self.__pulse_width_min[1]
            self.__degree[1] = self.__min_degree[1]
        elif self.__y_pulse > self.__pulse_width_max[1]:
            self.__y_pulse = self.__pulse_width_max[1]
            self.__degree[1] = self.__max_degree[1]

    # ========== function for rotating =======================================
    def rotate_by_angle(self, x_degree = None, y_degree = None):
        """
        An angle can be passed by which the playing field should be rotated.

        Parameters
        ----------
        x_degree : float, optional
            Number of degrees that the servo should rotate in x direction
        y_degree : float, optional
            Number of degrees that the servo should rotate in y direction

        Returns
        -------

        """
        if x_degree != None:
            self.__degree[0] += x_degree
            self.__calc_x_degree_to_pulse()
            channel = self.__channel[0]
        if y_degree != None:
            self.__degree[1] += y_degree
            self.__calc_y_degree_to_pulse()
            channel = self.__channel[1]

        data = self.__write_pulse(channel)
        return data

    # -------------------------------------------------------------------------
    def rotate_to_angle(self, x_degree = None, y_degree = None):
        """
        An angle can be passed to which the playing field should be rotated.

        Parameters
        ----------
        x_degree : float, optional
            Number of degrees at which the servo should rotate in x direction
        y_degree : float, optional
            Number of degrees at which the servo should rotate in y direction

        Returns
        -------

        """
        if x_degree != None:
            self.__degree[0] = x_degree
            self.__calc_x_degree_to_pulse()
            channel = self.__channel[0]
        elif y_degree != None:
            self.__degree[1] = y_degree
            self.__calc_y_degree_to_pulse()
            channel = self.__channel[1]

        data = self.__write_pulse(channel)
        return data

# -----------------------------------------------------------------------------
# Main (sample)
# -----------------------------------------------------------------------------

if __name__ == '__main__':
    servo = ServoCommunication()

    while True:
        #numx = float(input("Enter x degree: ")) # Taking input from user
        #value = servo.rotate_to_angle(x_degree= numx)
        #value = servo.rotate_by_angle(x_degree= numx)
        time.sleep(3)
        value = servo.rotate_to_angle(x_degree=2)
        print(value)
        time.sleep(3)
        value = servo.rotate_to_angle(y_degree=-4)
        print(value)
        time.sleep(3)
        value = servo.rotate_to_angle(x_degree=0)
        print(value)
        time.sleep(5)
        value = servo.rotate_to_angle(y_degree=0)
        print(value)
        time.sleep(5)