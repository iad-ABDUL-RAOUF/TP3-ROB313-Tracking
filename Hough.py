import numpy as np
import cv2
from math import pi, floor

roi_defined = False

# cap = cv2.VideoCapture('../Test-Videos/VOT-Ball.mp4')
# cap = cv2.VideoCapture('../Test-Videos/VOT-Basket.mp4')
# cap = cv2.VideoCapture('../Test-Videos/VOT-Car.mp4')
# cap = cv2.VideoCapture('../Test-Videos/VOT-Sunshade.mp4')
cap = cv2.VideoCapture('../Test-Videos/VOT-Woman.mp4')
# cap = cv2.VideoCapture('../Test-Videos/Antoine_Mug.mp4')
# cap = cv2.VideoCapture(0) #camera

nb_lines = 180 #nb de lignes dans la matrice R
threshold = 0.0001 #threshold of gradient for a pixel to be taken into account


def define_ROI(event, x, y, flags, param):
	global r,c,w,h,roi_defined
	# if the left mouse button was clicked, 
	# record the starting ROI coordinates 
	if event == cv2.EVENT_LBUTTONDOWN:
		r, c = x, y
		roi_defined = False
	# if the left mouse button was released,
	# record the ROI coordinates and dimensions
	elif event == cv2.EVENT_LBUTTONUP:
		r2, c2 = x, y
		h = abs(r2-r)
		w = abs(c2-c)
		r = min(r,r2)
		c = min(c,c2)  
		roi_defined = True

# take first frame of the video
ret,frame = cap.read()
size = frame.shape[0:2]
print("size = ", size)
# load the image, clone it, and setup the mouse callback function
clone = frame.copy()
cv2.namedWindow("First image")
cv2.setMouseCallback("First image", define_ROI)

while True:
	# display the image and wait for a keypress
	cv2.imshow("First image", frame)
	key = cv2.waitKey(1) & 0xFF
	# if the ROI is defined, draw it!
	if (roi_defined):
		# draw a green rectangle around the region of interest
		cv2.rectangle(frame, (r,c), (r+h,c+w), (0, 255, 0), 2)
	# else reset the image...
	else:
		frame = clone.copy()
	# if the 'q' key is pressed, break from the loop
	if key == ord("q"):
		break

print("Start computing")

track_window = (r,c,h,w)
# set up the ROI for tracking
roi = frame[c:c+w, r:r+h]
gray_roi =  cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
Ix = cv2.Sobel(gray_roi, -1, 1, 0, ksize=3)
Iy = cv2.Sobel(gray_roi, -1, 0, 1, ksize=3)
normGrad_roi = np.sqrt(Ix*Ix+Iy*Iy)
argGrad_roi = np.arctan2(Iy,Ix)

while ret:

    #Initializing the R table
    R = [];
    for i in range(nb_lines):
        R.append([]);
    #Filling the R table
    for i in range(w):
        for j in range(h):
            if normGrad_roi[i,j] > threshold:
                k = floor((argGrad_roi[i,j]+pi)*(nb_lines/(2*pi)))
                R[k].append([w,h]) # the "center" of the ROI is in top-right corner

    #Read the next frame
    ret,frame = cap.read()
    if ret:
        #Computing the norm and argument of the gradient
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        Ix = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
        Iy = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
        normGrad = np.sqrt(Ix*Ix+Iy*Iy)
        argGrad = np.arctan2(Iy,Ix)
        
        argDisplay = np.float32(argGrad.copy())
        argDisplay = np.floor((argDisplay+pi)*(255/(2*pi)))
        print("####################################################################################################################################################################")
        print("max = ",argDisplay.max())
        print("min = ",argDisplay.min())
        argDisplay = cv2.cvtColor(argDisplay,cv2.COLOR_GRAY2RGB)
        cv2.imshow('Argument of gradient',argDisplay)
        #Initializing the matrix for the votes
        weights = np.zeros(size)
        #Computing the votes for the position of the "center" of roi
        for i in range(size[0]):
            for j in range(size[1]):
                if normGrad[i,j] > threshold:
                    index = floor((argGrad[i,j]+pi)*(nb_lines/(2*pi)))
                    for point in R[index]:
                        if 0<= i+point[0] <size[0] and 0<= j+point[1]<size[1]:
                            weights[i+point[0],j+point[1]] += 1
                else:
                    argDisplay = cv2.circle(argDisplay, (i,j), 1, (0,0,255), -1)
        pos_roi = np.argwhere(weights==weights.max())[0]

        #Display arg
        cv2.imshow('Argument of gradient',argDisplay)

        #Updating ROI
        gray_roi = image[pos_roi[0]:pos_roi[0]+w, pos_roi[1]:pos_roi[1]+h]
        normGrad_roi = normGrad[pos_roi[0]:pos_roi[0]+w, pos_roi[1]:pos_roi[1]+h]
        argGrad_roi = argGrad[pos_roi[0]:pos_roi[0]+w, pos_roi[1]:pos_roi[1]+h]


            

    
