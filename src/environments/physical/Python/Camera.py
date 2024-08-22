"""
Class for capturing frames from a video camera.

@author: Marc Hensel
@contact: http://www.haw-hamburg.de/marc-hensel

@copyright: 2024, Marc Hensel
@version: 2024.01.25
@license: CC BY-NC-SA 4.0, see https://creativecommons.org/licenses/by-nc-sa/4.0/deed.en
"""

import cv2

class Camera():
    
    # -------------------------------------------------------------------------
    # Constructor
    # -------------------------------------------------------------------------
    
    def __init__(self, cameraID=1):
        """
        Constructor, tries to connect to a camera.
        
        The camera identifier is specified by the system. For a laptop,
        typically the build in camera has the identifier 0 and a camera
        connected e.g. by USB has the identifier 1.
        
        Parameters
        ----------
        cameraID : int, optional
            Camera to connect to (Default: 1)

        Returns
        -------
        None.

        """
        # Connect to camera
        print('Connecting to camera {}'.format(cameraID))
        self.__camera = cv2.VideoCapture(cameraID)

        # Get frame dimensions
        if self.__camera.isOpened():
            self.width  = self.__camera.get(cv2.CAP_PROP_FRAME_WIDTH)
            self.height = self.__camera.get(cv2.CAP_PROP_FRAME_HEIGHT)
        else:
            print('WARNING: Could not open camera.')
    
    # -------------------------------------------------------------------------
    # Release resources
    # -------------------------------------------------------------------------

    def close(self):
        """
        Closes the connection to the camera (i.e., releases the camera).

        Returns
        -------
        None.

        """
        self.__camera.release()
    
    # -------------------------------------------------------------------------
    # Get frame
    # -------------------------------------------------------------------------
        
    def nextFrame(self):
        """
        Get the next frame from the camera.

        Returns
        -------
        numpy.ndarray
            Copy of captured image ("frame").

        """
        # Is camera ready?
        if self.__camera.isOpened() is False:
            print('WARNING: Camera not ready.')
            return None
    
        # Read frame from camera
        isSuccess, frame = self.__camera.read()
        if isSuccess is False:
            print('WARNING: Could not read frame from camera')
            return None
        
        return frame.copy()
