import numpy as np
import cv2
import itertools

roi_defined = False

# cap = cv2.VideoCapture('../Test-Videos/VOT-Ball.mp4')
# cap = cv2.VideoCapture('../Test-Videos/VOT-Basket.mp4')
# cap = cv2.VideoCapture('../Test-Videos/VOT-Car.mp4')
# cap = cv2.VideoCapture('../Test-Videos/VOT-Sunshade.mp4')
# cap = cv2.VideoCapture('../Test-Videos/VOT-Woman.mp4')
cap = cv2.VideoCapture('../Test-Videos/Antoine_Mug.mp4')
# cap = cv2.VideoCapture(0) #camera

#madeleine
#figPath = '../'
#iad
figPath = '../newFig/'

nb_lines = 180 #nb de lignes dans la matrice R
threshold = 40 #threshold of gradient for a pixel to be taken into account


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
gray_roi =  cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
Ix = cv2.Sobel(gray_roi, cv2.CV_64F, 1, 0, ksize=3)
Iy = cv2.Sobel(gray_roi, cv2.CV_64F, 0, 1, ksize=3)
normGrad_roi = np.sqrt(Ix*Ix+Iy*Iy)
argGrad_roi = np.arctan2(Iy,Ix)

# contstruction des vecteurs de positions relative des pixels par rapport centre du template
roi_origin = np.array([w//2,h//2])
wToRoiOrigin, hToRoiOrigin = np.meshgrid(np.arange(0,w), np.arange(0,h), indexing = 'ij')
wToRoiOrigin = roi_origin[0] - wToRoiOrigin
hToRoiOrigin = roi_origin[1] - hToRoiOrigin
vectToRoiOrigin = np.concatenate((wToRoiOrigin.reshape((w,h,1)),
                                  hToRoiOrigin.reshape((w,h,1))),
                                 axis = 2)

# matrice des coordonnées des pixels des frames M[i,j,0] = i, M[i,j,1] = j
pixelIcoordinate, pixelJcoordinate = np.meshgrid(np.arange(0,size[0]), np.arange(0,size[1]), indexing = 'ij')
pixelIJcoordinate = np.concatenate((pixelIcoordinate.reshape((size[0],size[1],1)),
                                  pixelJcoordinate.reshape((size[0],size[1],1))),
                                 axis = 2)

# computing the R-table
# init R-table
R = []
# find meanigfull structures in ROI
isNormBigEnough_roi = normGrad_roi>threshold
# keep only values associated to meaningfull structures
argGrad_roi = argGrad_roi[isNormBigEnough_roi]
vectToRoiOrigin = vectToRoiOrigin[isNormBigEnough_roi]
# file in R-table lines
rTableIndex_roi = np.floor((argGrad_roi + np.pi)*(nb_lines/(2*np.pi)))
for i in range(nb_lines):
    R.append(vectToRoiOrigin[rTableIndex_roi == i]) 

cpt = 1
while ret:
    #Read the next frame
    ret,frame = cap.read()
    if ret:
        #Computing the norm and argument of the gradient
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        Ix = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
        Iy = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
        normGrad = np.sqrt(Ix*Ix+Iy*Iy)
        argGrad = np.arctan2(Iy,Ix)
        
        # store gradient argument to display
        argDisplay = argGrad.copy()
        argDisplay = np.uint8(np.floor((argDisplay+np.pi)*(255/(2*np.pi))))
        argDisplay = cv2.cvtColor(argDisplay,cv2.COLOR_GRAY2BGR)
        
        # find meanigfull structures
        isNormBigEnough = normGrad>threshold
        
        # put in red not meaningful structures
        argDisplay[~isNormBigEnough] = [0,0,255]
        
        # keep only values associated to meaningfull structures
        argGrad = argGrad[isNormBigEnough]
        rTableIndex = np.floor((argGrad + np.pi)*(nb_lines/(2*np.pi)))
        pixelIJcoordinate_bigEnough = pixelIJcoordinate[isNormBigEnough]
        
        #Initializing the matrix for the votes
        weights = np.zeros(size)
        
        #Computing the votes for the position of the "center" of roi
        for i in range(nb_lines): 
            for pixel,tableVect in itertools.product(pixelIJcoordinate_bigEnough[rTableIndex == i], R[i]):
                centerI = pixel[0]+tableVect[0]
                centerJ = pixel[1]+tableVect[1]
                if ((-1 < centerI) and (centerI < size[0]) and (-1 < centerJ) and (centerJ < size[1])):
                    weights[centerI,centerJ] = weights[centerI,centerJ] + 1
        
        #Find the best origin of tracked ROI
        pos_roi = np.unravel_index(np.argmax(weights), weights.shape)
        
        # Draw a blue rectangle on the current image
        r,c,h,w = pos_roi[1] - roi_origin[1], pos_roi[0] - roi_origin[0], h, w
        frame_tracked = cv2.rectangle(frame, (r,c), (r+h,c+w), (255,0,0) ,2)
        
        #Display arg
        cv2.imshow('vote',np.uint8(255*(weights/weights.max())))
        cv2.imshow('Sequence',frame_tracked)
        cv2.imshow('Argument of gradient',argDisplay)
        cv2.imshow('norm gradient',np.uint8(255*(normGrad/normGrad.max())))
        
        k = cv2.waitKey(60) & 0xff
        if k == 27:
            break
        elif k == ord('s'):
            cv2.imwrite(figPath+'Frame_%04d.png'%cpt,frame_tracked)
            cv2.imwrite(figPath+'vote_%04d.png'%cpt,weights)
            cv2.imwrite(figPath+'argGrad_%04d.png'%cpt,argDisplay)
            cv2.imwrite(figPath+'normGrad_%04d.png'%cpt,normGrad)
        cpt += 1
    else:
        break

cv2.destroyAllWindows()
cap.release()
            

    
