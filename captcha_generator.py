from captcha.image import ImageCaptcha
import string 
import random
import matplotlib.pyplot as plt
import numpy as np

def generate(w,h,num_char):
    char = string.digits + string.ascii_uppercase  # To create characters list
    random_string = ''.join(random.sample(char,num_char))
    image=ImageCaptcha(width=w,height=h).generate_image(random_string)
    return (image,random_string)

directory = '/home/ninjakx/Documents/opencv_works/'
labels = []

for i in range(10):
    img,char = generate(140,80,4)
    plt.imsave(directory+char+".png",np.array(img))
   

    
