"""
AI labyrinth main application.

@author: Marc Hensel
@contact: http://www.haw-hamburg.de/marc-hensel

@copyright: 2024, Marc Hensel
@version: 2024.07.16
@license: CC BY-NC-SA 4.0, see https://creativecommons.org/licenses/by-nc-sa/4.0/deed.en
"""

import cv2
import time
from Imaging import Imaging
from GUI import GUI

class App():
    
    # ----------------------------------------------------------------------
    # Constructor
    # ----------------------------------------------------------------------
        
    def __init__(self, cameraID=1, fps=10, isShowImageAnalysis=True):
        """
        Init application.

        Parameters
        ----------
        cameraID : int, optional
            Camera to open; see documentation of class Camera. (Default: 1)
            
        fps : int, optional
            Target frame rate. (Default: 10 frames per second)
            
        isShowImageAnalysis : boolean, optional
            Shows intermediate images if True. (Default: True)

        Returns
        -------
        None.

        """
        self.__framePeriodMs = int(1000 / fps)
        self.imaging = Imaging(cameraID)
        self.gui = GUI(self)

        self.__isShowImageAnalysis = isShowImageAnalysis
        self.__ballCenter = [-1,-1]
        self.__ballRadius = 0
        
    # ----------------------------------------------------------------------
    # Release resources
    # ----------------------------------------------------------------------

    def close(self):
        """
        Free allocated and reserved resources.

        Returns
        -------
        None.

        """
        self.imaging.close()
        self.gui.close()

    # ----------------------------------------------------------------------
    # Main application loop
    # ----------------------------------------------------------------------

    def run(self):
        """
        Main application loop.

        Returns
        -------
        None.

        """
        waitTimeMs = 1
        
        while cv2.waitKey(int(waitTimeMs)) != 27:
            # Measure time
            startTime = time.time()
            
            # Get and process next camera frame
            image, ballCenter, ballRadius = self.__processNextFrame()

            if ballCenter is not None:
                print('Object at ({:5.1f}, {:5.1f}) with r = {:4.1f}'.format(ballCenter[0], ballCenter[1], ballRadius))
                
            # Wait time depending on time elapsed (=> Establish frame rate).
            # Time must be >= 1 so that cv2.waitKey() reacts to user input.
            timeElapsedMs = 1000 * (time.time() - startTime)
            waitTimeMs = max(self.__framePeriodMs - timeElapsedMs, 1)

    # ----------------------------------------------------------------------
    # Imaging
    # ----------------------------------------------------------------------

    def __processNextFrame(self):
        """
        Get and analyse next image.
        
        If exactly one blob is identified as ball, its center and radius are returned.

        Returns
        -------
        image : numpy.ndarray
            Image containing the labyrinth, only.
        ballCenter : float[2]
            Blob's center location (x,y) or None.
        ballRadius : TYPE
            Blob's radius or 0.0.

        """
        # Get next camera frame and extracted pinball area
        frame, image = self.imaging.nextFrame()
        ballCenter = None
        ballRadius = 0.0
        
        if frame is not None:
            # Process new image
            binImage, ballCenter, ballRadius = self.imaging.detectBall()
            
            # Add annotations
            if ballCenter is not None:
                self.__ballCenter = ballCenter
                self.__ballRadius = ballRadius
                color = (0,255,0)
            else:
                color = (0,0,255)
            image = self.imaging.drawCircle(image, self.__ballCenter, self.__ballRadius + 1, color, 1)
            
            # Display images
            self.gui.showImages(frame, image)
            if self.__isShowImageAnalysis:
                self.gui.showImage('Binary image', binImage)
            
        return image, ballCenter, ballRadius

# ========== Main =============================================================

if __name__ == '__main__':
    app = App(cameraID=1, fps=10, isShowImageAnalysis=True)
    app.run()
    app.close()
