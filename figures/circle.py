import numpy as np
import cv2 as cv
 
# (x, y) : Centre coordinate
# r = Radius
# Color : Color in BGR
# Thickness : Thickness of stroke 

img = np.zeros((512,512,3), np.uint8)

x = 100
y = 100

r=40

Color = (255,255,255)
Thickness = 4

img = cv.circle(img, (x,y), r, Color, Thickness)
 
cv.imshow('Circle',img)
cv.waitKey(0)
