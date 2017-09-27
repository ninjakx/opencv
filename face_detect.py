import cv2
import numpy as np
face_detector=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
cap=cv2.VideoCapture(0)
while(1):
    retval,im=cap.read()
    gray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(gray,1.3,5)

    for (x,y,w,h) in faces:
        cv2.rectangle(im,(x,y),(x+w,y+h),(255,0,0),2)
    cv2.imshow('frame',im)
    c = cv2.waitKey(1)
    if c == 27: #Press esc to stop
        break

cap.release()
cv2.destroyAllWindows()

