"""
Image processing for AI labyrinth application.

@author: Marc Hensel, Sandra Lassahn
@contact: http://www.haw-hamburg.de/marc-hensel

@copyright: 2024, Marc Hensel
@version: 2024.07.16
@license: CC BY-NC-SA 4.0, see https://creativecommons.org/licenses/by-nc-sa/4.0/deed.en
"""

import cv2
import numpy as np
from ImageAcquisition import ImageAcquisition

class Imaging():
    
    # ----------------------------------------------------------------------
    # Constructor and initialization
    # ----------------------------------------------------------------------
        
    def __init__(self, cameraID):
        """
        Init image acquisition and image processing.

        Parameters
        ----------
        cameraID : int
            Camera to connect to. See documentation of class Camera.

        Returns
        -------
        None.

        """
        # Init image source
        self.acquisition = ImageAcquisition(cameraID=cameraID)

        # Init processing units
        self.threshold = 125
        self.__initBlobDetector()
        
        # Get first images
        self.frame, self.image = self.nextFrame()

    # ----------------------------------------------------------------------
        
    def __initBlobDetector(self, minArea=100, maxArea=1_750, minCircularity=0.6, maxCircularity=1.0):
        """
        Init blob detector to detect moving ball in binary difference images.
        
        Estimation of the areas in pixel, with 1 mm = 3 pixel:
            Area_ball = pi * (12.8 / 2)^2 mm^2 ~ 129 mm^2
            Area_ball = 129 (3 * 3 * pixel) ~ 1,161 pixel
            Area_center = pi * (5.0 / 2)^2 mm^2 ~ 20 mm^2
            Area_center = 20 (3 * 3 * pixel) ~ 180 pixel

        Parameters
        ----------
        minArea : int, optional
            Minimum area of a blob. (Default: 100)
        maxArea : int, optional
            Maximum area of a blob. (Default: 1,750)
        minCircularity : float, optional
            Minimum roundness of a blob. (Default: 0.6)
        maxCircularity : float, optional
            Maximum roundness of a blob. (Default: 1.0)

        Returns
        -------
        None.

        """
        params = cv2.SimpleBlobDetector_Params()
        params.filterByColor = False  # color filter used manually
        params.filterByArea = True
        params.minArea = minArea
        params.maxArea = maxArea
        params.filterByCircularity = True
        params.minCircularity = minCircularity
        params.maxCircularity = maxCircularity
        params.filterByConvexity = False
        params.filterByInertia = False
        self.__blobDetector = cv2.SimpleBlobDetector_create(params)

    # ----------------------------------------------------------------------
    # Release resources
    # ----------------------------------------------------------------------

    def close(self):
        """
        Release resources.

        Returns
        -------
        None.

        """
        self.acquisition.close()
        
    # ----------------------------------------------------------------------
    # Get next frame
    # ----------------------------------------------------------------------

    def nextFrame(self):
        """
        Get next camera frame and image of extracted labyrinth area.

        Returns
        -------
        numpy.ndarray
            Camera frame.
        numpy.ndarray
            Image of extracted labyrinth area.

        """
        self.frame, self.image = self.acquisition.nextFrame()
        return self.frame, self.image

    # ----------------------------------------------------------------------
    # Image processing and analysis
    # ----------------------------------------------------------------------

    def detectBall(self):
        """
        Process and analyze current image.
        
        Creates a binary image of positive differences and detects blobs.
        If only one blob is detected, its center location and radius are
        returned.

        Returns
        -------
        binImage : numpy.ndarray
            Binary difference image with detected blobs marked in color.
        ballCenter : [float, float]
            Blob's center location, if only one blob is detected (or None).
        ballRadius : float
            Blob's radius, if only one blob is detected (or None).

        """
        # Convert image to HSV color space
        hsvImage = cv2.cvtColor(self.image, cv2.COLOR_BGR2HSV)

        # Define color range for green (based on ball color)
        lower_green = np.array([40, 40, 40])
        upper_green = np.array([90, 255, 255])

        # Create a mask for green color
        mask = cv2.inRange(hsvImage, lower_green, upper_green)
        result = cv2.bitwise_and(self.image, self.image, mask=mask)

        # Convert result to grayscale for further processing
        grayResult = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)

        # Apply blob detection on the masked green image
        binImage, ballCenter, ballRadius = self.__blobDetection(grayResult)

        return binImage, ballCenter, ballRadius

    # ----------------------------------------------------------------------
    # NOT USED Now
    def __thresholdAndDilate(self, srcImage):
        """
        Apply threshold and dilate blobs.
        
        The structure element self.__dilateStructureElement is used for dilation.

        Parameters
        ----------
        srcImage : numpy.ndarray
            Gray-valued source to process.

        Returns
        -------
        numpy.ndarray
            Image after thresholding and morphological dilation.

        """
        # Create binary image
        _, binImage = cv2.threshold(srcImage, self.threshold, 255, cv2.THRESH_BINARY_INV)

        # Remove circles (thin lines)
        h_x = cv2.getStructuringElement(cv2.MORPH_RECT, (5,1)) 
        h_y = cv2.getStructuringElement(cv2.MORPH_RECT, (1,5)) 
        binImage = cv2.morphologyEx(binImage, cv2.MORPH_OPEN, h_x)
        binImage = cv2.morphologyEx(binImage, cv2.MORPH_OPEN, h_y)

        # Fill gaps
        h_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15,15)) 
        binImage = cv2.morphologyEx(binImage, cv2.MORPH_CLOSE, h_close)

        return binImage

    # ----------------------------------------------------------------------
    # Blob detection on filtered image
    # ----------------------------------------------------------------------

    def __blobDetection(self, binImage):
        """
        Detect blobs in binary image.
        
        The method uses the detector self.__blobDetector. Detected blobs are
        drawn into the binary image in color. If only one blob is detected,
        the method returns its center location and radius.

        Parameters
        ----------
        binImage : numpy.ndarray
            Binary image to analyze.

        Returns
        -------
        binImage : numpy.ndarray
            Binary source image with detected blobs marked in color.
        [float, float]
            Blob's center location (or None).
        float
            Blob's radius (or None).

        """
        # Detect blobs
        keypoints = self.__blobDetector.detect(binImage)

        # Draw detected blobs in images
        binImage = cv2.cvtColor(binImage, cv2.COLOR_GRAY2BGR)
        binImage = cv2.drawKeypoints(binImage, keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        
        # Determine keypoint representing moving ball
        if len(keypoints) > 0:
            maxKeypoint = keypoints[0]
            for key in keypoints:
                if key.size > maxKeypoint.size:
                    maxKeypoint = key
            return binImage, maxKeypoint.pt, (maxKeypoint.size + 1)/2
        else:
            return binImage, None, None


    # ----------------------------------------------------------------------
    # Drawing
    # ----------------------------------------------------------------------
        
    def drawCircle(self, image, center, radius, color, thickness):
        """
        Draw a circle in an image.

        Parameters
        ----------
        image : numpy.ndarray
            Image to draw circle in.
        center : float[2]
            Circle's center location (x,y).
        radius : float
            Circle's radius.
        color : int[3]
            Color formatted as BGR.
        thickness : int
            Line thickness.

        Returns
        -------
        numpy.ndarray
            Image with circle drawn.

        """
        return cv2.circle(image, (int(center[0]), int(center[1])), int(radius), color, thickness)
