"""
GUI of the application to train and solve virtual and physical labyrinths.

@author: Marc Hensel
@contact: http://www.haw-hamburg.de/marc-hensel
@copyright: 2024
@version: 2024.08.23
@license: CC BY-NC-SA 4.0, see https://creativecommons.org/licenses/by-nc-sa/4.0/deed.en
"""
import tkinter as tk
import webbrowser
import threading
from PIL import ImageTk, Image

class LabyrinthGUI():

    # =========================================================================
    # ========== Constructor ==================================================
    # =========================================================================

    def __init__(self, parentApp):
        """
        Constructor.
        
        Parameters
        ----------
        parentApp : WinderApp
            Application object to use to process user input.

        Returns
        -------
        None.

        """
        # Set parent application object
        self.parentApp = parentApp
        
        # GUI root frame and variables
        root = tk.Tk()
        root.title('Deep Labyrinth')
        self.__bg_color =  root.cget('bg')
        self.__isCounterClockwiseValue = tk.BooleanVar()       # State of the checkbox "Rotate counter-clockwise"

        # Create left (counter, stepper, and info) and right (speed control) GUI frames
        leftFrame = tk.Frame(root)
        self.__addCounterFrame(parent=leftFrame, padding=10)
        self.__addStepperMotorFrame(parent=leftFrame, padding=10)
        self.__addInfoFrame(parent=leftFrame, padding=10)
        self.__addImage(parent=leftFrame, dy=16)        
        rightFrame = self.__createRightFrame(parent=root, padding=10)

        # Layout and start GUI loop
        leftFrame.pack(side='left', anchor='n', padx=5, pady=5)
        rightFrame.pack(side='left', padx=5, pady=5)
        root.mainloop()
                
    # -------------------------------------------------------------------------
    
    def __addCounterFrame(self, parent, padding):
        """
        Add a frame containing the counter display and reset.

        Parameters
        ----------
        parent : tkinter.Frame
            GUI object to place created frame in.
        padding : int
            Space (padding) inside the frame boarder.

        Returns
        -------
        None.

        """
        # Create widgets
        frame = tk.LabelFrame(parent, text='Counter', padx=padding, pady=padding)
        self.__counterLabel = tk.Label(frame, text='0', width=5, height=1, background='black', foreground='white', font=('Arial', '53'))
        resetButton = tk.Button(frame, text='Reset', width=15, command=self.__onResetCounter)
        
        # Layout widgets
        self.__counterLabel.pack()
        resetButton.pack()
        frame.pack(side='top')
        
    # -------------------------------------------------------------------------
    
    def __addStepperMotorFrame(self, parent, padding):
        """
        Add a frame containing a stepper motor section to set the turning direction of the motor.

        Parameters
        ----------
        parent : tkinter.Frame
            GUI object to place created frame in.
        padding : int
            Space (padding) inside the frame boarder.

        Returns
        -------
        None.

        """
        frame = tk.LabelFrame(parent, text='Stepper motor', padx=padding, pady=padding)
        self.__directionCheckbox = tk.Checkbutton(frame, text='Rotate counter-clockwise', variable=self.__isCounterClockwiseValue, command=self.__onSetStepperDirection)
        self.__directionCheckbox.pack(anchor='w')
        frame.pack(side='top', anchor='w', fill='x')
        
    # -------------------------------------------------------------------------
    
    def __addInfoFrame(self, parent, padding):
        """
        Add a frame containing a hyperlink to the project's GitHub repository.

        Parameters
        ----------
        parent : tkinter.Frame
            GUI object to place created frame in.
        padding : int
            Space (padding) inside the frame boarder.

        Returns
        -------
        None.

        """
        frame = tk.LabelFrame(parent, text='Info', padx=padding, pady=padding)
        linkLabel = tk.Label(frame, text='Project on GitHub', anchor='w', font=('Artial 9'), fg="blue", cursor="hand2")
        linkLabel.bind("<Button-1>", lambda e: self.__onLink("https://github.com/MarcOnTheMoon/hexaphonic_guitar"))
        linkLabel.pack(side='top', anchor='w')
        frame.pack(side='top', anchor='w', fill='x')
        
    # -------------------------------------------------------------------------
    
    def __addImage(self, parent, dy=16):
        """
        Add the HAW Hamburg logo as image.

        Parameters
        ----------
        parent : tkinter.Frame
            GUI object to place created frame in.
        padding : int
            Space (padding) inside the frame boarder.

        Returns
        -------
        None.

        """
        # Image does not show if not stored as attribute using 'self.'
        self.image = ImageTk.PhotoImage(Image.open("images/HAW-160x50.png"))
        canvas = tk.Canvas(parent, width=self.image.width(), height=self.image.height() + dy)
        canvas.create_image(0, dy, anchor='nw', image=self.image)
        canvas.pack(side='top', anchor='w')
        
    # -------------------------------------------------------------------------
    
    def __createRightFrame(self, parent, padding):
        """
        Create a frame containing a slider to set the stepper motor speed and start/stop button.

        Parameters
        ----------
        parent : tkinter.Frame
            GUI object to place created frame in.
        padding : int
            Space (padding) inside the frame boarder.

        Returns
        -------
        None.

        """
        frame = tk.LabelFrame(parent, text='Speed [rps]', padx=padding, pady=padding)
        speedScale = tk.Scale(frame, width=100, length=280, from_=8, to=0, resolution=1, tickinterval=1, activebackground='red', command=self.__onSpeed)
        self.__startStopButton= tk.Button(frame, text='Start', height=2, activebackground='red', command=self.__onStartStop)
        speedScale.pack()
        self.__startStopButton.pack(fill='x')
        return frame
    
    # =========================================================================
    # ========== Callback methods =============================================
    # =========================================================================

    def __onUpdateCounter(self):
        """ Time callback method to update the counter.
        
        Queries the Arduino's counter value and updates the display. If the
        stepper motor is enabled (i. e., it might be turning), it starts a
        thread to call this update method again after a specific time period.

        Returns
        -------
        None.

        """
        if self.parentApp != None:
            # Query and update counter
            count = self.parentApp.getRevCount()
            self.__counterLabel.config(text = str(count))            
            
            # Call update again when stepper is enabled (else it does not move)
            isEnabled = (self.__startStopButton.cget('text') == 'Stop')
            if isEnabled == True:
                threading.Timer(2.0, self.__onUpdateCounter).start()
        else:
            print('Update count (no app connected)')

    # -------------------------------------------------------------------------

    def __onResetCounter(self):
        """ Button callback method to reset the counter to 0.

        Returns
        -------
        None.

        """        
        if self.parentApp != None:
            self.__counterLabel.config(text = '0')
            self.parentApp.resetRevCounter()
        else:
            print('Counter reset (no app connected)')

    # -------------------------------------------------------------------------
    
    def __onSetStepperDirection(self):
        """ Checkbox callback method to set the stepper motor's direction of rotation.

        Returns
        -------
        None.

        """        
        if self.parentApp != None:
            isClockwise = (self.__isCounterClockwiseValue.get() == False)
            self.parentApp.setDirection(isClockwise=isClockwise)
        else:
            print('Rotate counter-clockwise: {} (no app connected)'.format(self.__isCounterClockwiseValue.get()))

    # -------------------------------------------------------------------------
    
    def __onStartStop(self):
        """ Button callback method to start/stop (i.e., enable/disable) the stepper motor.
        
        Starts a thread to frequently query the Arduino's counter and update
        the counter display. The thread ends automatically when "enable" is
        set to False.

        Returns
        -------
        None.

        """        
        # Toggle button text (start/stop)
        isStart = (self.__startStopButton.cget('text') == 'Start')
        if isStart == True:
            self.__startStopButton.config(text='Stop', bg='red')
        else:
            self.__startStopButton.config(text='Start', bg=self.__bg_color)

        # Start or stop the stepper motor        
        if self.parentApp != None:
            self.parentApp.enableMotor(isEnabled=isStart)
            threading.Timer(0.25, self.__onUpdateCounter).start()
        else:
            print('Enable stepper: {} (no app connected)'.format(isStart))

    # -------------------------------------------------------------------------
    
    def __onSpeed(self, value):
        """ Slider (scale) callback method to set the stepper motor's speed.

        Parameters
        ----------
        value : int
            New slider value (i.e., the target speed in revolutions per second).

        Returns
        -------
        None.

        """        
        if self.parentApp != None:
            self.parentApp.setSpeed(revsPerSec=int(value))
        else:
            print('Set speed: {} (no app connected)'.format(value))

    # -------------------------------------------------------------------------
    
    def __onLink(self, url):
        """ Hyperlink callback method to the GitHub page in a webbrowser.

        Parameters
        ----------
        url : string
            Internet URL to open in the browser.

        Returns
        -------
        None.

        """        
        webbrowser.open_new(url)
    
# -----------------------------------------------------------------------------
# Main (sample)
# -----------------------------------------------------------------------------

if __name__ == '__main__':
    LabyrinthGUI(parentApp=None)
