import numpy as np
import cv2 as cv
 
# (x1, y1) : Vertex at the top left
# (x2, y2) : Lower right vertex (x, y)
# Color : Color in BGR
# Thickness : Thickness of stroke 

img = np.zeros((512,512,3), np.uint8)

x1 = 100
y1 = 100

x2 = 400
y2 = 400

Color = (255,255,255)
Thickness = 4


img = cv.rectangle(img, (x1,y1), (x2,y2), Color, Thickness)
 
cv.imshow('rectangle',img)
cv.waitKey(0)
