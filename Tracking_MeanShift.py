import numpy as np
import cv2

roi_defined = False
 
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

#cap = cv2.VideoCapture('../Test-Videos/VOT-Ball.mp4')
# cap = cv2.VideoCapture('../Test-Videos/VOT-Basket.mp4')
# cap = cv2.VideoCapture('../Test-Videos/VOT-Car.mp4')
# cap = cv2.VideoCapture('../Test-Videos/VOT-Sunshade.mp4')
# cap = cv2.VideoCapture('../Test-Videos/VOT-Woman.mp4')
cap = cv2.VideoCapture('../Test-Videos/Antoine_Mug.mp4')
# cap = cv2.VideoCapture(0) #camera

# take first frame of the video
ret,frame = cap.read()
# load the image, clone it, and setup the mouse callback function
clone = frame.copy()
cv2.namedWindow("First image")
cv2.setMouseCallback("First image", define_ROI)

#madeleine
#figPath = '../'
#iad
figPath = '../newFig/'

#parameters
histUpdtRate = 0.2
chooseHist = 'teinte' # 'grad', 'teinte'


# keep looping until the 'q' key is pressed
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

    



track_window = (r,c,h,w)


if chooseHist == 'teinte':
    # set up the ROI for tracking
    roi = frame[c:c+w, r:r+h]
    # conversion to Hue-Saturation-Value space
    # 0 < H < 180 ; 0 < S < 255 ; 0 < V < 255
    hsv_roi =  cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    # computation mask of the histogram:
    # Pixels with S<30, V<20 or V>235 are ignored 
    mask = cv2.inRange(hsv_roi, np.array((0.,30.,20.)), np.array((180.,255.,235.)))
    # Marginal histogram of the Hue component
    roi_hist = cv2.calcHist([hsv_roi],[0],mask,[180],[0,180])

if chooseHist == 'grad':
    # calcul de l'histogramme de l'argument du gradient
    grayFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    grayRoi = grayFrame[c:c+w, r:r+h]
    Ix_roi = cv2.Sobel(grayRoi, cv2.CV_64F, 1, 0, ksize=3)
    Iy_roi = cv2.Sobel(grayRoi, cv2.CV_64F, 0, 1, ksize=3)
    normGrad_roi = np.float32(np.sqrt(Ix_roi*Ix_roi+Iy_roi*Iy_roi))
    argGrad_roi = np.float32(np.arctan2(Iy_roi,Ix_roi))
    mask = cv2.inRange(normGrad_roi, 30, 1000000)
    roi_hist = cv2.calcHist([argGrad_roi],[0],mask,[180],[-np.pi,np.pi])

# Histogram values are normalised to [0,255]
cv2.normalize(roi_hist,roi_hist,0,255,cv2.NORM_MINMAX)


# Setup the termination criteria: either 10 iterations,
# or move by less than 1 pixel
term_crit = ( cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1 )

cpt = 1
while(1):
    ret ,frame = cap.read()
    if ret == True:
        
        if chooseHist == 'teinte':
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    		# Backproject the model histogram roi_hist onto the 
    		# current image hsv, i.e. dst(x,y) = roi_hist(hsv(0,x,y))
            dst = cv2.calcBackProject([hsv],[0],roi_hist,[0,180],1)

        if chooseHist == 'grad':
            grayFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            Ix = cv2.Sobel(grayFrame, cv2.CV_64F, 1, 0, ksize=3)
            Iy = cv2.Sobel(grayFrame, cv2.CV_64F, 0, 1, ksize=3)
            argGrad = np.float32(np.arctan2(Iy,Ix))
            dst = np.uint8(cv2.calcBackProject([argGrad],[0],roi_hist,[-np.pi,np.pi],1))        
        
        cv2.imshow('Poids',dst)
        # apply meanshift to dst to get the new location
        ret, track_window = cv2.meanShift(dst, track_window, term_crit)

        # Draw a blue rectangle on the current image
        r,c,h,w = track_window
        frame_tracked = cv2.rectangle(frame, (r,c), (r+h,c+w), (255,0,0) ,2)
        cv2.imshow('Sequence',frame_tracked)

        #update histogram template
        if chooseHist == 'teinte':
            hsv_roi = hsv[c:c+w, r:r+h]
            mask = cv2.inRange(hsv_roi, np.array((0.,30.,20.)), np.array((180.,255.,235.)))
            new_roi_hist = cv2.calcHist([hsv_roi],[0],mask,[180],[0,180])
        if chooseHist == 'grad': # TODO grad version
            normGrad_roi = np.float32(np.sqrt(Ix[c:c+w, r:r+h]*Ix[c:c+w, r:r+h]+Iy[c:c+w, r:r+h]*Iy[c:c+w, r:r+h]))
            argGrad_roi = argGrad[c:c+w, r:r+h]
            mask = cv2.inRange(normGrad_roi, 30, 1000000)
            new_roi_hist = cv2.calcHist([argGrad_roi],[0],mask,[180],[-np.pi,np.pi])
        cv2.normalize(new_roi_hist,new_roi_hist,0,255,cv2.NORM_MINMAX)
        roi_hist = (1-histUpdtRate)*roi_hist+histUpdtRate*new_roi_hist
        
        k = cv2.waitKey(60) & 0xff
        if k == 27:
            break
        elif k == ord('s'):
            cv2.imwrite(figPath+'Frame_%04d.png'%cpt,frame_tracked)
            cv2.imwrite(figPath+'dst_%04d.png'%cpt,dst)
        cpt += 1
    else:
        break

cv2.destroyAllWindows()
cap.release()
