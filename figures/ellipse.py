import numpy as np
import cv2 as cv
 
# (x, y) : Centre coordinate
# (h, w) : Length of minor and major axes
# r = Radius
# SA = starting angle (calculated clockwise)
# RA =starting angle (calculated clockwise)
# FA = final angle (calculated clockwise)
# Color : Color in BGR
# Thickness : Thickness of stroke 

x, y = 300, 150
h, w = 100, 50
RA = 45
SA = 130
FA = 270
Color = (255,255,255)
Thickness = 4

img = np.zeros((512,512,3), np.uint8)

img = cv.ellipse(img, (x, y), (h, w), RA, SA, FA, Color, Thickness)

cv.imshow('Ellipse',img)
cv.waitKey(0)
