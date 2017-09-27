import numpy as np
import cv2 as cv
 
# (x1, y1) : Starting point of coordinates 
# (x2, y2) : End point of coordinates
# Color : Color in BGR
# Thickness : Thickness of stroke 

img = np.zeros((512,512,3), np.uint8)

x1 = 100
y1 = 100

x2 = 400
y2 = 400

Color = (255,255,255)
Thickness = 4


img = cv.line(img, (x1,y1), (x2,y2), Color, Thickness)
 
cv.imshow('Line',img)
cv.waitKey(0)
