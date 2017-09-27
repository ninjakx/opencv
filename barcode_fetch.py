from imutils import contours
import numpy as np
import argparse
import imutils
import cv2

ap = argparse.ArgumentParser()
ap.add_argument("-i1", "--image", required = True, help = "Path to the image")
args = vars(ap.parse_args())
 
image = cv2.imread(args["image"])

gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

#Scharr gradient magnitude 
gradx = cv2.Sobel(gray, ddepth = cv2.CV_32F, dx = 1, dy = 0, ksize = -1)
grady = cv2.Sobel(gray, ddepth = cv2.CV_32F, dx = 0, dy =1, ksize = -1)

gradient = cv2.subtract(gradx,grady)
gradient = cv2.convertScaleAbs(gradient)

#cv2.imshow('image',gradient)
#cv2.waitKey(0)

blurred = cv2.blur(gradient,(9,9))
_, thresh = cv2.threshold(blurred,225,255,cv2.THRESH_BINARY)

#cv2.imshow('image',thresh)
#cv2.waitKey(0)

kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (21,7))
closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

#cv2.imshow('image',closed)
#cv2.waitKey(0)

closed = cv2.erode(closed, None,iterations = 4)
closed = cv2.dilate(closed, None,iterations = 4)

#cv2.imshow('image',closed)
#cv2.waitKey(0)

(_,cnts, _) = cv2.findContours(closed.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

area = sorted(cnts , key = cv2.contourArea, reverse = True)[0]


rect = cv2.minAreaRect(area)
box = np.int0(cv2.boxPoints(rect))

cv2.drawContours(image, [box], -1, (0,255,0) , 1)
cv2.imshow("image", image)
cv2.waitKey(0)



