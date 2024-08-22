"""
The desired angular position can be transmitted to the Arduino via a serial interface.

@authors: Sandra Lassahn
@contact: http://www.haw-hamburg.de/marc-hensel
@copyright: 2024
@version: 2024.04.25
@license: CC BY-NC-SA 4.0, see https://creativecommons.org/licenses/by-nc-sa/4.0/deed.en
"""

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
        self.__degree = [0.0, 0.0]

    # ========== transmit message to the Arduino =========================================
    def __write_angle(self, channel, degree):
        """
        Combines all data into a string for the message and transmits it to the Arduino via the serial interface.
        sent data: channel and degree for one servo

        Parameters
        ----------
        channel: int

        Returns
        -------
        String

        """
        command = str(channel) + ";" + str(degree)
        print(command)
        self.__arduino.writeData(command)
        time.sleep(0.05)
        data = self.__arduino.readLine()
        return data

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
        String

        """
        if x_degree != None:
            self.__degree[0] += x_degree
            data = self.__write_angle(self.__channel[0], self.__degree[0])
        if y_degree != None:
            self.__degree[1] += y_degree
            data = self.__write_angle(self.__channel[1], self.__degree[1])
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
        String

        """
        if x_degree != None:
            self.__degree[0] = x_degree
            data = self.__write_angle(self.__channel[0], self.__degree[0])
        elif y_degree != None:
            self.__degree[1] = y_degree
            data = self.__write_angle(self.__channel[1], self.__degree[1])

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