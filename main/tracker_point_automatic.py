import cv2
import numpy as np
import json
from toposmooth import topology_preserving_smooth, load_image, asftmed
from PIL import Image
import numpy as np

# I'm going to try to implement another algorithmto track the wavefront
# I'll try tracking white shapes instead of dark ones, , by binarizing the image ( this worked and it's more robust, it's the one implemented)
# quick function to double the size of an image

def double_size_image(image):
    height, width = image.shape[:2]
    new_height, new_width = height * 2, width * 2
    resized_image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
    return resized_image


def binarize( gray, threshold):
    # Simple binary thresholding
    binary = np.zeros_like(gray)
    binary[gray >= threshold] = 255
    binary[gray < threshold] = 0
    return binary

def get_line ( Image, pt,L): # gets the line centered at pt  of length L, Image is already binarized
    x=int(pt[0])
    y=int(pt[1])
    half_L=int(L/2)
    
    line=Image[y,x - half_L : x + half_L +1]
    
    return line

def get_white_st_pt_line(line):
    # gets the list of indexes of first white pxls of each white region on the line
    indexes=[]
    n=len(line)
    i=0
    was_black=True 
    while i<n:
        color=line[i]
        if color==255:
            if was_black:
                indexes.append(i)
            was_black=False
        elif color==0:
            was_black=True
        i+=1
    return indexes

def get_black_st_pt_line(line):
    #returns the list of indexes of first black pxls of each black string on the line, I have an idea about how to avoid counting several strings in one shape
    # when they regroup because of oscillation : for each black , we count the number of white shapes around it in a small rectangle that cover each edge. For 1 string there'll be 2
    # for 2 there'll be 4, 6 for 3, etc...
    # Then once we know how many strings really hide in the black shape, we divide the black lin by this number to get the actual centre of each string.qqqq 
    indexes=[]
    n=len(line)
    i=0
    was_white=True
    count_b=0 
    while i<n:
        color=line[i]
        if color==0:
            if was_white:
                indexes.append(i)
            was_white=False
        elif color==255:
            was_white=True
        i+=1
    return indexes

def get_list_st_pts(Image, pt,L): # gets the list of first white pxls on the line centered at pt of length L
    line=get_line(Image,pt,L)
    indexes=get_white_st_pt_line(line)
    
    st_pts=[]
    half_L=int(L/2)
    for index in indexes:

        st_pts.append((pt[0] - half_L + index, pt[1]))
    return st_pts

def neighbors_f(pt,Image):# gets the 4 adjacent pts of , keeping in mind the image's shape
    
    H,W= Image.shape
    
    neigh=[]
    if pt[1]+1<H : #each time we check whether we're not out of range
        neigh.append(( pt[0],pt[1]+1))
    if pt[1]-1>=0:
        neigh.append((pt[0],pt[1]-1))
    if pt[0]+1<W:
        neigh.append(( pt[0]+1,pt[1]))
    if pt[0]-1>=0:
        neigh.append((pt[0]-1,pt[1] ))
    return neigh

def neighbors_e(pt,Image):# gets the 8 adjacent pts of , keeping in mind the image's shape
    
    H,W= Image.shape
    
    neigh=[]
    if pt[1]+1<H : #each time we check whether we're not out of range
        neigh.append(( pt[0],pt[1]+1))
    if pt[1]-1>=0:
        neigh.append((pt[0],pt[1]-1))
    if pt[0]+1<W:
        neigh.append(( pt[0]+1,pt[1]))
    if pt[0]-1>=0:
        neigh.append((pt[0]-1,pt[1] ))
    if pt[1]+1<H and pt[0]+1<W:
        neigh.append(( pt[0]+1,pt[1]+1))
    if pt[1]+1<H and pt[0]-1>=0:
        neigh.append(( pt[0]-1,pt[1]+1))
    if pt[1]-1>=0 and pt[0]+1<W:
        neigh.append(( pt[0]+1,pt[1]-1))
    if pt[1]-1>=0 and pt[0]-1>=0:
        neigh.append(( pt[0]-1,pt[1]-1))
    return neigh

def get_shape(Image, start_pt,N): #basic exploration algorithm in a grid, Image already binarized
    #N is 4 or 8 depending on the connectivity we want
    shape_pts=[]
    to_visit=[start_pt] # The pile containing the pts to be searched through
    while len(to_visit)>0:
        pt=to_visit.pop(0)
        shape_pts.append(pt)
        
        neigh= neighbors_f(pt,Image) if N==4 else neighbors_e(pt,Image)
       

        for neighbor in neigh:
            if neighbor not in shape_pts and neighbor not in to_visit: #unchecked pt
                if Image[neighbor[1], neighbor[0]]==255:
                    to_visit.append(neighbor) # we add the neighbor to the pile
    return shape_pts 

def get_b_shape(Image,start_pt,N): # same algorithm qbut for black shapes
    shape_pts=[]
    
    to_visit=[start_pt] # The pile containing the pts to be searched through
    while len(to_visit)>0:
        pt=to_visit.pop(0)
        shape_pts.append(pt)
        neigh= neighbors_f(pt,Image) if N==4 else neighbors_e(pt,Image)      
        for neighbor in neigh:
            if neighbor not in shape_pts and neighbor not in to_visit: #unchecked pt
                if Image[neighbor[1], neighbor[0]]==0:
                    to_visit.append(neighbor) # we add the neighbor to the pile
    return shape_pts 


def get_corners( shape):
    #returns a list [ min_x, max_x, min_y, max_y ], just useful to get dimensions laters for the points we're interested in
    Ly, Lx = [], []
    for pt in shape:
        Lx.append(pt[0])
        Ly.append(pt[1])
    min_x = min(Lx)
    max_x = max(Lx)
    min_y = min(Ly)
    max_y = max(Ly)
    return [min_x, max_x, min_y, max_y]
def get_height( shape): #returns the height of the shape
    corners=get_corners(shape)
    height=corners[3]-corners[2]+1
    
    return height
def get_width( shape): #returns the width of the shape
    corners=get_corners(shape)
    width=corners[1]-corners[0]+1
    return width
def get_shapes_list(Image,st_pt,L ): # gets the line centered at st_pt, the white shapes around st_pt, and their sizes
    st_pts=get_list_st_pts(Image,st_pt,L)
    shapes=[]
    for pt in st_pts:
        shape=get_shape(Image, pt,4) # we use 4-connectivity here
        shapes.append(shape)
        
    return shapes

def get_last_big_shape(Image, st_pt,L): # using the fact that on the videos after the tangled point white shapes are 2 times-ish smaller than the ones 
  
    shapes=get_shapes_list(Image,st_pt,L)
    
    if len(shapes)==0:
        return None
    heigths=[]
    for shape in shapes:
        heigths.append( get_height(shape))
    max_height=max(heigths)
    threshold=max_height/2
    i=0
    while i<len(heigths) and heigths[i]>threshold :
        i+=1
    if i==0:
        return shapes[0]
    else:
        return shapes[i-1] 

def get_rightest_edge(shape): # gets the wavefront, we define as the rightest point of the shape
    L= get_corners(shape)
    max_x= -1
    rightest_pt=None
    for pt in shape:
        if pt[0]>max_x:
            max_x=pt[0]
            rightest_pt=pt
    return rightest_pt # we got the wavefront point

#finally the function iterating the search for the wave point
def get_wave_pt(gray,threshold, st_pt, L):
    Image= binarize(gray, threshold)
    shape =get_last_big_shape(Image, st_pt,L)
    if shape is None:
        return None
    wave_pt=get_rightest_edge(shape)
    return wave_pt


### Functions to try to track the released strings

def get_pseudo_line(wavefront, origin) -> list: #renvoie une ligne de points
    # renvoie une liste des points d'une pseudo-ligne tracée entre les deux points
    Lx = wavefront[0]-origin[0]
    Ly= wavefront[1]-origin[1]
    line=[]
    a=0
    if Ly ==0:
        a = Lx
    else:
        a= Lx//Ly   
    for i in range(Lx):
        line.append(( origin[0]+i,origin[1]+i//a))
    line.append(wavefront)
    return line 

def get_all_b_st_pts(Image, wavefront, origin): #Image must be in shades of gray, and binarized, returns the black starting points of the pseudo line, which we can maybe consider to be 
    # the oscillating springs. returns the indexes
    line = get_pseudo_line(wavefront,origin)
    colors=[]
    for pt in line:
   
        colors.append(Image[pt[1],pt[0]])
        
    indexes= get_black_st_pt_line(colors)
    
    return indexes

def get_centers ( st_i, S,l): #S number of springs in sub black line starting at starting index st_i. colors is a line [x in {0,255}]
    #l is the length of the subline starting at i
    subcenters=[]
    a = l//S
    i = st_i
    for j in range( S):
        subcenters.append(int(i+a/2+a*j))
    #we now have a list of the centers ( rounded up) of each subline
    

    return subcenters 



#all that's left to do is, for a given sub_line of our pseudo_line, to make a function that tells us for sure how many springs are in it

#I now have almost all the "centers" of the springs in each frame, but one problem remain: during the oscillations that follow the untangling
# the springs touch their neighbors and are recognized as only one spring for sometimes two or 3. To solve this, i think for each black shape i detect with the previous algorith,
# I could look for each separate white shape in a rectangle around it, of height the height of the wave -2/4 pixels  ( the margin is narrow because if we detect the white rectangles of other rows we're screwed)
# the number W of white shapes gives me the number S of strings in the shape with S= W/2.
# then if we have L the length of the black line detected on the pseudo line, we can divide the line in S sub lines and assign the center of each sub-line  as centre of each one of the S springs

def get_string_n(line, st_i,H, image): #line the list of points of the pseudoline , H the height of the rectangle in which we'll search for white shapes ( can vary between videos)
    #will also return the length of the subline
    l=0
    i=st_i
    
    while i<len(line) and image[line[i][1],line[i][0]]==0:
        
        l+=1
        i+=1
    xs= line[st_i][0]-3
    xf= line[st_i+l-1][0]+4
    ys= line[st_i][1]-H
    yf= line[st_i][1]+H+1
    region = image[ys:yf,xs:xf]
    
    
    #we got the relevant rectangle, now we implement an algorithm counting the connex white shapes ( 4 -connectivity) ( update: we should use 8-connectivity)
    N=0 # number of strings detected in the subline starting at st_i
    indices = list(np.ndindex(region.shape))
    

    while len(indices)>0:
        pt= indices.pop()
        if region[pt[0],pt[1]]==255: # if the black point
            N+=1
            shape= get_shape(region,(pt[1],pt[0]),8) # indexes a reversed compared to the order i use in get_b_shape( x,y)=> (y,x)
            
            

            for point in shape:
                if (point[1],point[0]) in indices:
                    indices.remove((point[1],point[0])) # the list is cleared of all the points of a potential white shape
            
        
    if N%2 ==1:
        N+=1 # so that N/2 is an integer
    return int(N/2), l
    
### The sometimes whimsical state of the binarized image leads the algorithm to sometimes detect too many White-shapes,
# so i'd like to try another approach.
# Basically, I think that if the pseudo-line connecting the origin to the center conatins black sublines that correspond to several strings, if we offset it vertically, we should in the 
#end get a line that crosses each string only once. The thing is, we have to guess the offset

def offset_line(line, offset): #offset is positive or negative integer
    offset_line=[]
    for pt in line:
        offset_line.append( ( pt[0], pt[1]+offset) ) # we offset in the y direction so first component of our points
    return offset_line

def count_black_shapes(line, Image): # counts the number of black sublines on the line of length at least 2
    colors=[]
    
    for pt in line:
   
        colors.append(Image[pt[1],pt[0]])
        
    indexes= get_black_st_pt_line(colors)
    for i in indexes:
        # we remove the indexes that correspond to black shapes of length 1
        if i+1 < len(colors):
            if colors[i+1]==255:
                indexes.remove(i)
    
    return len(indexes)

#now we need a function that decides when to stop trying offsets without knowing max_offset
def find_best_offset(line, Image):
    base_ct= count_black_shapes(line, Image)
    offset=0
    best_offset=0
    current_ct= base_ct

    for i in range(10): # we try 20 offsets in each direction
        offset=i
        offsetted_line_a= offset_line(line, offset)
        offsetted_line_b= offset_line(line, -offset)
        ct_b= count_black_shapes( offsetted_line_b, Image)
        if ct_b > current_ct: # means we found a better offsetand some bundled strings
            best_offset= -offset
            current_ct= ct_b
        ct_a= count_black_shapes(offsetted_line_a, Image)
        if ct_a > current_ct: # means we found a better offsetand some bundled strings can now be told apart
            best_offset= offset
            current_ct= ct_a
        
        # so basically this loop has to stop at the edges of the wave
        

    return offset_line(line, best_offset) # now we have a line which should have each string alone



### début du wave point tracker d'Honoré

class WavePointTracker :
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

        # wave points ( {frame_idx : (x, y)})
        self.wave_points = {}
        self.string_points=[] # Where we'll try to see the oscillations of each loose string

        # tracking parameters, to be modified when the tracking fails
        self.tracking_enabled = False
        self.string_tracking_enabled = False # to be activated only when the tracked straing won't bundle anymore
        self.search_width = 10  # dimensions of the search region
        self.search_heigth = 5
        self.triangle_search = False # determines whether the search is conducted in a triangular region in front of the last wavefront or not
        self.R_line= 20  # Radius of the line on which we search for the starting points of the right shapes
        self.threshold = 110 # threshold for binarization, the right value depends on the video, between 100 and 150 is the usual range
        self.white_shapes= [] # The list of white shapes' centers already detected'
        self.average_shape_per_search=5 # number of shapes we have to  look at in white_shapes to be sure we're not adding a shape that's already in it after an iteration
        self.wave_height = 6 # wave half-height, used in the detection of each loose string. Can be had with a simple print for each viedo in f° get_height. I usually choose a little lower half the heights, to be sure not to detect other rows
        self.mouse_x = 0
        self.mouse_y = 0

    
    def mouse_callback(self, event, x, y, flags, param):
        """
        Left-click: set origin (first) or wave front position (subsequent)
        Right-click: track a single string at current frame
        """
        self.mouse_x = x
        self.mouse_y = y

        if event == cv2.EVENT_LBUTTONDOWN : 
            if self.origin_point is None :
                self.origin_point = (x, y)
                print(f'Origin set at frame {self.current_frame_idx} : ({x}, {y})')
            else :
                self.wave_points[self.current_frame_idx] = (x, y)
                print(f'Frame {self.current_frame_idx} : {x}, {y}')

        

    def mouse_callbackb(self, event, x, y, flags, param):
        """
        Right-click: track a single string at current frame
        """
        self.mouse_x = x
        self.mouse_y = y

        if event == cv2.EVENT_RBUTTONDOWN : 
            
            self.string_points.append([self.current_frame_idx , (x, y)])     
    def update_white_shapes(self,Image,shapes):
        #shapes is the list of white shapes detected in the last frame, we search in the N last shapes if they are already in self.white_shapes, if not we add them
        for shape in shapes:
            corners = get_corners(shape)
            center= ( int((corners[0]+corners[1])/2), int((corners[2]+corners[3])/2) )
            # we have the central point of the shape, now we check if this point appears in the last N shapes of self.white_shapes
            is_new=True
            for prev_center in self.white_shapes[-self.average_shape_per_search:]:
                prev_shape= get_shape(Image, prev_center) # ofc this isn't right cause Image isn't defined here
                for pt in prev_shape: 
                    if pt==center:
                        is_new=False
                        break
                if not is_new:
                    break
            if is_new:
                self.white_shapes.append(center)
        return


    def find_next_point(self, frame, center_x, center_y, search_width, search_heigth):
        # we look for the rightest point of the last "big" white shape around the previous wave front, just like the eye would do
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        wavefront= get_wave_pt(gray, self.threshold, (center_x, center_y),self.R_line)
        
        if wavefront is not None:
            
            return wavefront
        else:
            return center_x, center_y
        
        
    def track_wave_front(self, frame) :
        """
        Automated tracking. 
        """
        
        
        if not self.tracking_enabled or not self.wave_points :
            return None
        
        # find most recent tracked position
        last_tracked_frame = max([f for f in self.wave_points.keys() if f < self.current_frame_idx ], default=None)
        if last_tracked_frame is None:
            return None
        last_x, last_y = self.wave_points[last_tracked_frame]

        x_new, y_new = self.find_next_point(frame, last_x, last_y,  self.search_width, self.search_heigth)
       
        
        
        
        """
        # second idea using the hypothesis that even when some are bundled, there exists a line parallel to the  central line that contains each string seaparately
        line = get_pseudo_line((x_new,y_new),self.origin_point)
        best_line= find_best_offset(line, Image)
        print(count_black_shapes(best_line,Image))
        # problem: i still don't have a purely growing sequence, this approach still seems too sensitive
        for pt in best_line:
            cv2.circle(Image, (pt[0],pt[1]),1, (0,255,255),1)
           
        cv2.imshow('binarized',Image)
        """

        
        """
        part of the code trying to detect the oscillations with the topology algorithm (unefficient so far)
        bis= frame.copy()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        Image= binarize(gray,self.threshold)
        if  self.origin_point is not None:
             line= get_pseudo_line((x_new,y_new),self.origin_point)
             st_indexes = get_all_b_st_pts(Image,(x_new,y_new),self.origin_point)
             center_indexes=[]
             for i in st_indexes:
                px,py=line[i][0], line[i][1]
                
                
                S,l= get_string_n(line,i,8,Image)
                cv2.rectangle(bis, (px-3 , py - 8), (px + l +3, py + 8), (255,255,0), 1)
                subcenters = get_centers(i,S,l)
                center_indexes+= subcenters
             
             for i in center_indexes:
                 pt_x ,pt_y = line[i][0], line[i][1]
                 cv2.circle(bis,(pt_x,pt_y),2, (0,0,255), 1 )
        """
                
        #cv2.imshow('colo',bis)
        
        return (x_new, y_new)

    def get_next_string_point(self, frame):
        
        Image= binarize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY),self.threshold)
        #we assume there is already points in self.string_points
        if len(self.string_points)==0:
            print("No string point to base the tracking on")
            return None
        last_frame_idx, last_point= self.string_points[-1]
        last_point=(last_point[0]//2, last_point[1]//2)
        line =get_line(Image, last_point, 15)
        black_st_pts= get_black_st_pt_line(line)
        if len(black_st_pts)==0:
            return None
        else:
            max= 100
            best_i= black_st_pts[0]
            for i in black_st_pts:
                # we'll return the point closest to the center, we're in the case of small oscillations around the center
                if abs(i - 7)< max:
                    max= abs(i - 7)
                    best_i= i
            return (2*( last_point[0] -7 + best_i ), 2*last_point[1] )
    def create_trackbars(self):
        """
        Create trackbars.
        """
        cv2.namedWindow('Controls')
        cv2.createTrackbar('Frame', 'Controls', 0, self.frame_count - 1, lambda x : None)
    
    
    def vizualise(self, frame,st_tr):
        """est
        Generate annotated visualization:
        - Green: origin point ;
        - Red: wave front and trajectory ;
        - Blue : search region (when tracking enabled).
        """
        display = frame.copy()
        
        
        
        # origine point
        if self.origin_point is not None :
            ox, oy = self.origin_point
            cv2.circle(display, (ox, oy), 6, (0,255, 0), -1)
            cv2.line(display, (ox, 0), (ox, self.height), (255,0,0), 1, cv2.LINE_AA)

        # current wave front
        if self.current_frame_idx in self.wave_points : 
            wx, wy = self.wave_points[self.current_frame_idx]
            #cv2.circle(display, (wx, wy), 6, (0, 0, 255), -1)
            cv2.line(display, (wx, 0), (wx, self.height), (0, 0, 255), 1, cv2.LINE_AA)

            # search region visualization
            if self.tracking_enabled :
                self.search_width = int(self.search_width)
                self.search_heigth = int(self.search_heigth)

                #cv2.rectangle(display, (wx-self.R_line , wy - 2), (wx + self.R_line, wy + 2), (255,255,0), 1)

        # wave trajectory 
        sorted_frames = sorted([f for f in self.wave_points.keys() if f <= self.current_frame_idx])
        if len(sorted_frames) > 1 :
            for i in range (len(sorted_frames) - 1):
                f1, f2 = sorted_frames[i], sorted_frames[i+1]
                p1 = self.wave_points[f1]
                p2 = self.wave_points[f2]
                cv2.line(display, p1, p2, (0,0,255), 1, cv2.LINE_AA)
        # the tracked string in yellow
        if len(self.string_points)>0:
            
            sx, sy = self.string_points[-1][1]
            st_x, st_y = self.string_points[0][1]
            cv2.line(st_tr, (st_x, 0), (st_x, 2*self.height), (0, 0, 255), 2, cv2.LINE_AA)
            cv2.circle(display, (sx//2, sy//2), 2, (0, 255, 255), -1)
            cv2.circle(st_tr, (sx, sy), 4, (0, 0, 255), -1)


        info = f"Frame {self.current_frame_idx}/{self.frame_count-1}"
        cv2.putText(display, info, (10,25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
       
        return display, st_tr

    def run(self):
        """
        Main tracking interface.
        Controls: n/b (navigate), t (toggle tracking), s (save), q (quit)
        """
        self.create_trackbars()
        cv2.namedWindow('Wave tracker')
        cv2.setMouseCallback('Wave tracker', self.mouse_callback)
        cv2.namedWindow('String tracker')
        cv2.setMouseCallback('String tracker', self.mouse_callbackb)
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

            if not ret :
                print(f'Frame {self.current_frame_idx} cannot be read')
                break
            
            # Auto-track if enabled and frame not annotated
            if self.tracking_enabled and self.current_frame_idx not in self.wave_points :
                new_point = self.track_wave_front(frame)
                if new_point is not None:  
                        self.wave_points[self.current_frame_idx] = new_point
            if self.string_tracking_enabled and self.string_points[-1][0]!= self.current_frame_idx: # we assume the last frame had a string point
                new_point=self.get_next_string_point(frame)
                if new_point is not None:
                    self.string_points.append( [ self.current_frame_idx , new_point ] )
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            Image= binarize(gray,self.threshold)
            resized= double_size_image(Image)

            st_tr = cv2.cvtColor(resized, cv2.COLOR_GRAY2RGB)
            
            display, st_tr = self.vizualise(frame,st_tr)
            cv2.imshow('String tracker',st_tr)
            cv2.imshow('Wave tracker', display)
            

            key = cv2.waitKey(30) 

            if key == ord('q'):
                break
            elif key == ord('a'):
                self.string_tracking_enabled = not self.string_tracking_enabled
                print(f"String tracking: {'ON' if self.tracking_enabled else 'OFF'}")
            elif key == ord('c'):
                if len(self.string_points)>0:
                    print("Last string point removed")
                    #now we put this point in the first position
                    self.string_points.insert(0, self.string_points.pop() )
            elif key == ord('s'):
                self.save_data()
            elif key == ord('r'):
                 self.origin_point = None
            elif key == ord('t'):
                 self.tracking_enabled = not self.tracking_enabled
                 print(f"Tracking: {'ON' if self.tracking_enabled else 'OFF'}")
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

    def save_data(self):
            """
            Export tracking results to JSON.
            """
            data = {
                "video_path" : self.video_path,
                "fps" : self.fps,
                "frame count" : int(self.frame_count),
                "resolution" : {"width" : int(self.width), "height" : int(self.height)},
                "origin" : {
                    'x' : float(self.origin_point[0]),
                    'y' : float(self.origin_point[1])
                },
                "wave_front" : [],
                "tracked_string": []

            }

            # compute quantities and add points to wave_front
            for frame_idx in sorted(self.wave_points.keys()):
                x, y = self.wave_points[frame_idx]
                time_s = frame_idx / self.fps
                ox, oy = self.origin_point
                distance = abs(x - ox)

                data["wave_front"].append(
                    {
                        "frame" : int(frame_idx),
                        "time" : round(float(time_s), 4),
                        "x" : float(x),
                        "y" : float(y),
                        "distance from origin" : round(float(distance), 2)
                    }
                )
            for A in self.string_points:
                sx, sy = A[1]
                time_s = A[0] / self.fps
                ox, oy = self.origin_point
                distance = abs(sx - ox)//2

                data["tracked_string"].append(
                    {
                        "frame" : int(frame_idx),
                        "time" : round(float(time_s), 4),
                        "x" : float(sx),
                        "y" : float(sy),
                        "distance from origin" : round(float(distance), 2)
                    }
                )

             # write to JSON file    
            json_path = "results/wave_tracking.json"
            with open(json_path, "w") as f :
                json.dump(data, f, indent=2)

if __name__ == "__main__":
    
    video_path = "data/frame_v7.avi"
    tracker = WavePointTracker(video_path)
    tracker.run()
    