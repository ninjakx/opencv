import cv2

#Define this so that camera get some time to adjust itself 
frames = 30

camera = cv2.VideoCapture(0)
if not camera.isOpened():
    raise IOError("Cannot open webcam")

def get_image():
    retval, im = camera.read()
    return im

for i in range(frames):
    temp = get_image()
print("Capturing photo")
photo = get_image()
file = "/home/ninjakx/myphoto.png"

cv2.imwrite(file, photo)

del(camera)
