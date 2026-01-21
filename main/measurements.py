#file used to track some lengths
# Here 1px= 0.2mm
import cv2
import numpy as np
import json


class Measurer :
    """
    Semi-automated tracker for wave front propagation in video data.
    Combines manual annotation with automated tracking based on intensity minima.
    """

    def __init__(self, video_path):
        self.video_path = video_path
        self.cap = cv2.VideoCapture(video_path)
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # current frame
        self.current_frame_idx = 0
        
        # origin point 
        self.origin_point = None  # Reference point 

        self.points = []  

        
        
        

        # mouse state
        self.mouse_x = 0
        self.mouse_y = 0

        # video properties
        self.wave_height= 6
        
        

    
    def mouse_callback(self, event, x, y, flags, param):
        """
        Left-click: set origin (first) or wave front position (subsequent)
        Right-click: track a single string at current frame
        """
        self.mouse_x = x
        self.mouse_y = y

        if event == cv2.EVENT_LBUTTONDOWN : 
            self.points.append( (self.current_frame_idx , (x, y)) )
            if len(self.points) == 2 :
                distance=  (self.points[1][1][0]- self.points[0][1][0])**2 + (self.points[1][1][1]- self.points[0][1][1])**2 
                print(distance**0.5 )
                self.points.pop()
                self.points.pop()
    

      
    


    

    
    def create_trackbars(self):
        """
        Create trackbars.
        """
        cv2.namedWindow('Controls')
        cv2.createTrackbar('Frame', 'Controls', 0, self.frame_count - 1, lambda x : None)
    
    
    def vizualise(self, frame):
        """est
        Generate annotated visualization:
        - Green: origin point ;
        - Red: wave front and trajectory ;
        - Blue : search region (when tracking enabled).
        """
        display = frame.copy()
        
        for pt in self.points:
            frame_idx, (px, py) = pt
            if frame_idx == self.current_frame_idx:
                cv2.circle(display, (px, py), 5, (0, 255, 0), -1)  # origin point in green
            else:
                cv2.circle(display, (px, py), 3, (0, 0, 255), -1)  # wave front points in red
        
        
        
       
        return display

    def run(self):
        """
        Main tracking interface.
        Controls: n/b (navigate), t (toggle tracking), s (save), q (quit)
        """
        self.create_trackbars()
        cv2.namedWindow('Wave tracker')
        cv2.setMouseCallback('Wave tracker', self.mouse_callback)
        
        print("  LEFT CLICK  : Place origin (blue) or wave front (red)")
        print("  RIGHT CLICK : Track one string")
        print("  n        : Next frame")
        print("  b        : Previous frame")
        print("  r        : Reset origin")
        print("  t        : tracking on/off")
        print("  s        : Save data")
        print("  q        : Quit")
        print("  a        : Start autotracking of the strings")
        print()

        while True : 
            self.current_frame_idx = cv2.getTrackbarPos('Frame', 'Controls')
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame_idx)
            ret, frame = self.cap.read()

            
            
            display = self.vizualise(frame)
            
            cv2.imshow('Wave tracker', display)
            

            key = cv2.waitKey(30) 

            if key == ord('q'):
                break
            
                   
            
           
            elif key == ord('n'):
                if self.current_frame_idx < self.frame_count - 1 :
                    self.current_frame_idx += 1
                    cv2.setTrackbarPos('Frame', 'Controls', self.current_frame_idx)
            elif key == ord('b'):
                if self.current_frame_idx > 0 :
                    self.current_frame_idx -= 1
                    cv2.setTrackbarPos('Frame', 'Controls', self.current_frame_idx)

        self.cap.release()
        cv2.destroyAllWindows()

    

if __name__ == "__main__":
    
    video_path = "data/frame_v5.avi"
    tracker = Measurer(video_path)
    tracker.run()
    