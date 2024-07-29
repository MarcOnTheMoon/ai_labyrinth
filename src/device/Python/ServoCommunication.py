"""
The pulse width is determined from the desired angular positions according to
the used servos and transmitted to the Arduino via a serial interface.

In order to be able to establish serial communication with the Arduino install
Pyserial in Anaconda by the command 'conda install conda-forge::pyserial'.

@authors: Sandra Lassahn
@contact: http://www.haw-hamburg.de/marc-hensel
@copyright: 2024
@version: 2024.04.25
@license: CC BY-NC-SA 4.0, see https://creativecommons.org/licenses/by-nc-sa/4.0/deed.en
"""

import serial
import math
import time
class ServoCommunication:

    # ========== Constructor ==================================================
    def __init__(self, port=None):
        """
        Constructor.

        Parameters
        ----------
        port : String, optional
            Port number for serial communication to the arduino, default COM5

        Returns
        -------
        None.
        """

        if port is None:
            port = 'COM5'
        arduino = serial.Serial(port=port, baudrate=115200, timeout=.1)
        self.__arduino = arduino
        self.__x_channel = 1  #Connected channel of the servo to the PWM driver
        self.__x_pulse_width_max = 2650 #The maximum and minimum pulse width is different for each servo
        self.__x_pulse_width_min = 350
        self.__x_pulse_middle = (self.__x_pulse_width_max + self.__x_pulse_width_min) / 2
        self.__x_pulse_per_degree = (self.__x_pulse_width_max - self.__x_pulse_middle)/(math.asin(8.6/155.5)*180/math.pi)
        self.__x_degree = 0.0
        self.__x_min_degree = (self.__x_pulse_width_min - self.__x_pulse_middle) / self.__x_pulse_per_degree
        self.__x_max_degree = (self.__x_pulse_width_max - self.__x_pulse_middle) / self.__x_pulse_per_degree
        self.__y_channel = 0
        self.__y_pulse_width_max = 2540
        self.__y_pulse_width_min = 420
        self.__y_pulse_middle = (self.__y_pulse_width_max + self.__y_pulse_width_min) / 2
        self.__y_pulse_per_degree = (self.__y_pulse_width_max - self.__y_pulse_middle) / (math.asin(5.2 / 122.5) * 180 / math.pi)
        self.__y_degree = 0.0
        self.__y_min_degree = (self.__y_pulse_width_min - self.__y_pulse_middle) / self.__y_pulse_per_degree
        self.__y_max_degree = (self.__y_pulse_width_max - self.__y_pulse_middle) / self.__y_pulse_per_degree

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

        """
        if channel == self.__x_channel:
            command = str(channel) + ";" + str(self.__x_pulse)
        elif channel == self.__y_channel:
            command = str(channel) + ";" + str(self.__y_pulse)

        print(command)
        self.__arduino.write(bytes(command, 'ascii'))
        time.sleep(0.05)
        data = self.__arduino.readline().decode('ascii').strip()
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
        self.__x_pulse = int(self.__x_degree * self.__x_pulse_per_degree + self.__x_pulse_middle)
        #Check whether the calculated pulse width is within the permitted range, otherwise limit it
        if self.__x_pulse < self.__x_pulse_width_min:
            self.__x_pulse = self.__x_pulse_width_min
            self.__x_degree = self.__x_min_degree
        elif self.__x_pulse > self.__x_pulse_width_max:
            self.__x_pulse = self.__x_pulse_width_max
            self.__x_degree = self.__x_max_degree

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
        self.__y_pulse = int(self.__y_degree * self.__y_pulse_per_degree + self.__y_pulse_middle)
        # Check whether the calculated pulse width is within the permitted range, otherwise limit it
        if self.__y_pulse < self.__y_pulse_width_min:
            self.__y_pulse = self.__y_pulse_width_min
            self.__y_degree = self.__y_min_degree
        elif self.__y_pulse > self.__y_pulse_width_max:
            self.__y_pulse = self.__y_pulse_width_max
            self.__y_degree = self.__y_max_degree

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
            self.__x_degree += x_degree
            self.__calc_x_degree_to_pulse()
            channel = self.__x_channel
        if y_degree != None:
            self.__y_degree += y_degree
            self.__calc_y_degree_to_pulse()
            channel = self.__y_channel

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
            self.__x_degree = x_degree
            self.__calc_x_degree_to_pulse()
            channel = self.__x_channel
        elif y_degree != None:
            self.__y_degree = y_degree
            self.__calc_y_degree_to_pulse()
            channel = self.__y_channel

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