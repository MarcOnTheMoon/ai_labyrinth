"""
Graphical user interface for AI labyrinth.

@author: Marc Hensel
@contact: http://www.haw-hamburg.de/marc-hensel

@copyright: 2024, Marc Hensel
@version: 2024.07.16
@license: CC BY-NC-SA 4.0, see https://creativecommons.org/licenses/by-nc-sa/4.0/deed.en
"""

import cv2

class GUI():
    
    # ----------------------------------------------------------------------
    # Constructor
    # ----------------------------------------------------------------------
        
    def __init__(self, app):
        """
        Inits windows and event handling.

        Parameters
        ----------
        app : App
            Main application object (required for event handling).

        Returns
        -------
        None.

        """
        self.__app = app
        
        # Print usage hints
        print('\nUsage: Press <ESC> to quit')
        print('Usage: Left click camera frame to set corners of labyrinth\n')
        
        # Init windows and event handling
        cv2.namedWindow('Camera')
        cv2.namedWindow('Labyrinth')
        
        cv2.setMouseCallback('Camera', GUI.onMouseClick, app)
        cv2.createTrackbar('Threshold', 'Labyrinth', 0, 255, self.onTrackbarThreshold)
        cv2.setTrackbarPos('Threshold', 'Labyrinth', pos=app.imaging.threshold)

    # ----------------------------------------------------------------------
    # Release resources
    # ----------------------------------------------------------------------

    def close(self):
        """
        Close all OpenCV windows.

        Returns
        -------
        None.

        """
        cv2.destroyAllWindows()

    # ----------------------------------------------------------------------
    # Event handling
    # ----------------------------------------------------------------------
        
    def onMouseClick(event, x, y, flags, param):
        """
        Mouse event handling in window showing camera frame.
        
        Selects a new corner point of the ROI defining the labyrinth area.

        Parameters
        ----------
        event : TYPE
            Information on type of button event.
        x : int
            x-coordinate of the new point.
        y : int
            y-coordinate of the new point.
        flags : TYPE
            Not used.
        param : App
            Main application object, which will receive and process the event data.

        Returns
        -------
        None.

        """
        if event == cv2.EVENT_LBUTTONUP:
            param.imaging.acquisition.changeSourceCorner(x, y)
            
    # ----------------------------------------------------------------------

    def onTrackbarThreshold(self, value):
        """
        On trackbar change for threshold value.

        Parameters
        ----------
        value : int
            Trackbar value selected by the user.

        Returns
        -------
        None.

        """
        self.__app.imaging.threshold = value
                
    # ----------------------------------------------------------------------
    # Display data
    # ----------------------------------------------------------------------
        
    def showImages(self, cameraFrame, labyrinthImage):
        """
        Update displays of camera frame and image containing the labyrinth.

        Parameters
        ----------
        cameraFrame : numpy.ndarray
            Camera frame to show.
        labyrinthImage : numpy.ndarray
            Labyrinth image to show.

        Returns
        -------
        None.

        """
        cv2.imshow('Camera', cameraFrame)
        cv2.imshow('Labyrinth', labyrinthImage)

    # ----------------------------------------------------------------------

    def showImage(self, windowName, image):
        """
        Show an image in a named OpenCV window.

        Parameters
        ----------
        windowName : string
            Window name.
        image : numpy.ndarray
            Image to show.

        Returns
        -------
        None.

        """
        cv2.imshow(windowName, image)
