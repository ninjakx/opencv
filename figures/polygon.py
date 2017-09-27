import numpy as np
import cv2 as cv

# xi,yi : Array of coordinates 
# If_close = True, if it is a closed line
# Color : Color in BGR
# Thickness : Thickness of stroke 

img = np.zeros((512,512,3), np.uint8)
pts = np.array([[10,5],[20,130],[170,20],[150,110]], np.int32)
pts = pts.reshape((-1,1,2))
cv.polylines(img,[pts],True,(255,255,255))
cv.imshow('Polygon',img)
cv.waitKey(0)

