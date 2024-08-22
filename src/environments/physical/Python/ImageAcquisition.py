"""
Acquire image input data for AI labyrinth.

The data contains the original camera frame as well as the labyrinth area cut
out (i.e., mapped to an image).

@author: Marc Hensel, Sandra Lassahn
@contact: http://www.haw-hamburg.de/marc-hensel

@copyright: 2024, Marc Hensel
@version: 2024.07.15
@license: CC BY-NC-SA 4.0, see https://creativecommons.org/licenses/by-nc-sa/4.0/deed.en
"""

import cv2
import numpy as np
from Camera import Camera

class ImageAcquisition():

    # -------------------------------------------------------------------------
    # Constructor
    # -------------------------------------------------------------------------

    def __init__(self, cameraID=1, width=3*274, height=3*228, isRotate180=True):
        """
        Connects to a camera and gets the first pair of images.
        
        The images are the camera frame and an image of the pinball playing
        field cut from the frame. In this context, the mapping from the
        camera frame to the labyrinth area is initialized.
        
        The physical labyrinth has a dimension of 274x228 mm, which is mapped
        to 3*(274x228) pixels by default.

        Parameters
        ----------
        cameraID : int, optional
            Camera to connect to (Default: 1)
            
        width : int, optional
            Width of the image containing the labyrinth, only. (Default: 3*274)
            
        height : int, optional
            Height of the image containing the labyrinth, only. (Default: 3*228)
            
        isRotate180 : boolean, optional
            Rotate camera frame 180° left, if True. (Default: True)

        Returns
        -------
        None.

        """
        # Connect to camera
        self.__camera = Camera(cameraID=cameraID)

        # Set geometry of the image containing the pinball field
        self.__field = {
            'width': width,
            'height': height }

        # Set rotation
        self.__isRotate180 = isRotate180

        # Set initial region in camera frame regarded as labyrinth field
        boarder = 25
        self.__field['upper left'] = [boarder, boarder]
        self.__field['upper right'] = [self.__camera.width - boarder, boarder]
        self.__field['lower left'] = [boarder, self.__camera.height - boarder]
        self.__field['lower right'] = [self.__camera.width - boarder, self.__camera.height - boarder]

        # Calculate transformation to cut out labyrinth field
        self.__updatePerpectiveTransform()

        # Get first pair of images
        self.frame, self.fieldImage = self.nextFrame()

    # -------------------------------------------------------------------------
    # Release resources
    # -------------------------------------------------------------------------

    def close(self):
        """
        Close camera.

        Returns
        -------
        None.

        """
        self.__camera.close()

    # -------------------------------------------------------------------------
    # Change corner of ROI in source frame
    # -------------------------------------------------------------------------

    def changeSourceCorner(self, x, y):
        """
        Sets a new coordinate to select the labyrinth area.
        
        Set a new coordinate in the camera frame corresponding to one of the
        corners of the labyrinth. The method replaces the corner being closest
        to the new point (x,y).

        Parameters
        ----------
        x : int
            x-coordinate of the new corner point.
        y : int
            y-coordinate of the new corner point.

        Returns
        -------
        None.

        """
        labels = ['upper left', 'upper right', 'lower left', 'lower right']

        # Determine closest corner to new coordinates
        distances = [0., 0., 0., 0.]
        for i in range(4):
            corner = self.__field[labels[i]]
            distances[i] = (corner[0] - x)**2 + (corner[1] - y)**2
        minIndex = distances.index(min(distances))
        
        # Assign new coordinates to closest corner
        self.__field[labels[minIndex]] = [x, y]
        self.__updatePerpectiveTransform()

    # -------------------------------------------------------------------------
    # Perspective transform
    # -------------------------------------------------------------------------

    def __updatePerpectiveTransform(self):
        """
        Update the 3x3 matrix to map ROI in frame to image containing labyrinth
        field.

        Returns
        -------
        None.

        """
        # Corners of source ROI in camera frame
        srcCorners = np.float32([
            self.__field['upper left'],
            self.__field['upper right'],
            self.__field['lower left'],
            self.__field['lower right']])
        
        # Corners of destination ROI (i.e., image containing labyrinth field)
        dstCorners = np.float32([
            [0, 0],
            [self.__field['width'], 0],
            [0, self.__field['height']],
            [self.__field['width'], self.__field['height']]])
        
        # Calculate transformation matrix
        self.__transformMatrix = cv2.getPerspectiveTransform(srcCorners, dstCorners)

    # -------------------------------------------------------------------------
    # Get next pair of images
    # -------------------------------------------------------------------------
    
    def nextFrame(self, isDrawSourceROI = True):
        """
        Get the next camera frame and extracted labyrinth as image.

        Parameters
        ----------
        isDrawSourceROI : boolean, optional
            Draw ROI corresponding to labyrinth in frame, if true. (Default: True)

        Returns
        -------
        numpy.ndarray
            Camera frame (with labyrinth area marked, if isDrawSourceROI is True).
        numpy.ndarray
            Extracted labyrinth as image.

        """
        self.frame = self.__camera.nextFrame()
        
        # Apply rotation
        if self.__isRotate180 == True:
            self.frame = cv2.rotate(self.frame, cv2.ROTATE_180)
        
        # Get labyrinth area from camera frame
        self.fieldImage = cv2.warpPerspective(
            self.frame,
            self.__transformMatrix,
            (self.__field['width'], self.__field['height']))
        
        # Draw source ROI of labyrinth area in camera frame
        if isDrawSourceROI:
            points = np.array([
                self.__field['upper left'],
                self.__field['upper right'], 
                self.__field['lower right'],
                self.__field['lower left']],
                np.int32)
            cv2.polylines(self.frame, [points], True, (0,0,255))
        
        return self.frame, self.fieldImage

    def get_field(self):
        """
            Get field width and height

            Parameters
            ----------
            None

            Returns
            -------
            int
               Field width
            int
               Field height
        """
        #needed for calculating in the correct coordinate system
        return self.__field['width'], self.__field['height']