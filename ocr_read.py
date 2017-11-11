from PIL import Image
import pytesseract
import argparse
import cv2
import os
import numpy as np
import imutils
import io
from google.cloud import vision
from oauth2client.client import GoogleCredentials
credentials = GoogleCredentials.get_application_default()
from googleapiclient.discovery import build

service = build('compute', 'v1', credentials=credentials)


add = argparse.ArgumentParser()
add.add_argument("-i","--image",required=True,help="path to input image to be OCR 'd")
add.add_argument("-p", "--preprocess", type=str, default="thresh",
	help="type of preprocessing to be done")
args = vars(add.parse_args())
image = cv2.imread(args["image"])
gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

if args["preprocess"] == "thresh":
    gray = cv2.threshold(gray,0,255,cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)   

elif args["preprocess"] == "blur":
    gray = cv2.bilateralFilter(image,9,75,75)
    gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    
    

#rectKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 3))


filename="{}.png".format(os.getpid())
cv2.imwrite(filename,gray)

vision_client = vision.Client()

with io.open(filename, 'rb') as image_file:
    content = image_file.read()
    image = vision_client.image(content=content,)


#image = vision_client.image(content=content)

texts = image.detect_text()
print('Texts:')
print(texts[0].description)





#cv2.imshow("Image",image)
cv2.imshow("Output",gray)

#text = pytesseract.image_to_string(Image.open(filename))
os.remove(filename)
#print(text)
'''read_text='"{}"'.format(texts[0].description)
speech ="espeak -s 100 " +  read_text
os.system(speech)'''



cv2.waitKey(0)

# export GOOGLE_APPLICATION_CREDENTIALS=/home/ninjakx/Desktop/py-work/apikey.json

