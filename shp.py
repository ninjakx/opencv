import argparse 
import imutils 
import cv2
from collections import OrderedDict
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("-i","--image",required=True, help="path to image")

args = vars(ap.parse_args())

'''
        # define the list of boundaries
        boundaries = [
	    ([17, 15, 100], [50, 56, 200]),
	    ([86, 31, 4], [220, 88, 50]),
	    ([25, 146, 190], [62, 174, 250]),
	    ([103, 86, 65], [145, 133, 128])
        ]

        # loop over the boundaries
        for (lower, upper) in boundaries:
            # create NumPy arrays from the boundaries
            lower = np.array(lower, dtype = "uint8")
            upper = np.array(upper, dtype = "uint8")
 
            # find the colors within the specified boundaries and apply
            # the mask
            mask = cv2.inRange(image, lower, upper)
            print(mask)
            output = cv2.bitwise_and(image, image, mask = mask)
            print(output)
 
            # show the images
            cv2.imshow("images", np.hstack([image, output]))
            cv2.waitKey(0)
'''


class ColorLabeler:
    def __init__(self):
        # initialize the colors dictionary, containing the color
        # name as the key and the RGB tuple as the value

        colors = OrderedDict({
                 "red": (255, 0, 0),
                 "green": (0, 255, 0),
                 "blue": (0, 0, 255)})
 
        # allocate memory for the L*a*b* image, then initialize
        # the color names list
        self.lab = np.zeros((len(colors), 1, 3), dtype="uint8")
        self.colorNames = []
 
        # loop over the colors dictionary
        for (i, (name, rgb)) in enumerate(colors.items()):
            # update the L*a*b* array and the color names list
            self.lab[i] = rgb
            self.colorNames.append(name)
 
        # convert the L*a*b* array from the RGB color space
        # to L*a*b*
        self.lab = cv2.cvtColor(self.lab, cv2.COLOR_RGB2LAB)

    def label(self, image, c):
        # construct a mask for the contour, then compute the
        # average L*a*b* value for the masked region
        mask = np.zeros(image.shape[:2], dtype="uint8")
        cv2.drawContours(mask, [c], -1, 255, -1)
        mask = cv2.erode(mask, None, iterations=2)
        mean = cv2.mean(image, mask=mask)[:3]
 
        # initialize the minimum distance found thus far
        minDist = (np.inf, None)
 
        # loop over the known L*a*b* color values
        for (i, row) in enumerate(self.lab):
            # compute the distance between the current L*a*b*
            # color value and the mean of the image
            d = np.linalg.norm(row[0]-mean)
            #d = dist.euclidean(row[0], mean)
 
            # if the distance is smaller than the current distance,
            # then update the bookkeeping variable
            if d < minDist[0]:
                minDist = (d, i)
 
        # return the name of the color with the smallest distance
        return self.colorNames[minDist[1]]




class ShapeDetector:

 
    '''def order_points_old(pts):
        # initialize a list of coordinates that will be ordered
        # such that the first entry in the list is the top-left,
        # the second entry is the top-right, the third is the
        # bottom-right, and the fourth is the bottom-left
        rect = np.zeros((4, 2), dtype="float32")
 
        # the top-left point will have the smallest sum, whereas
        # the bottom-right point will have the largest sum
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]
 
        # now, compute the difference between the points, the
        # top-right point will have the smallest difference,
        # whereas the bottom-left will have the largest difference
        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]
 
        # return the ordered coordinates
        return rect'''
    def __init__(self):
        pass
 
    def detect(self, c):
        # initialize the shape name and approximate the contour
        shape = "unidentified"
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.04 * peri, True)

        # if the shape is a triangle, it will have 3 vertices

        if len(approx) == 3:
            shape = "triangle"
 
            # if the shape has 4 vertices, it is either a square or
            # a rectangle
        elif len(approx) == 4:
            pts = []
            # compute the bounding box of the contour and use the
            # bounding box to compute the aspect ratio
            x1,y1,w1,h1 = cv2.boundingRect(c)
            a,b,c,d = approx
            #box [[ 17 151]
            #[ 17  17]
            #[283  17]
            #[283 151]]

            #print(approx)
            #print(a,b,c,d)
            for i in a,b,c,d:
                pts.append((i.astype("int").tolist())[0])
            #print("pts",pts)  
            box = np.array(pts, dtype="int")  
            pts = box                   
            
            #print(x1,y1,w1,h1)
            cv2.rectangle(image,(x1,y1),(x1+w1,y1+h1),(255,255,0),2)
            y_axis=[]
            for y in a,b,c,d:
                y_axis.append(y[0][1])
            y_axis = sorted(y_axis)
            #print(y_axis)
           
            topLeftX = min(a[0][0],c[0][0]);

            topLeftY = min(a[0][1], b[0][1]);
            #print("t",topLeftX,topLeftY)
            width = max(b[0][0], d[0][0]) - topLeftX;
            height = max(c[0][1], d[0][1]) - topLeftY;
            #print(max(b[0][0], d[0][0]))
            #print(min(a[0][1], b[0][1]))
            #print(width,height)
            #dst = cv2.cornerHarris(gray,2,3,0.04)
            #print(dst)

            (x, y, w, h) = cv2.boundingRect(approx) 
            #print(x,y,w,h)
            #print(w,h)
            #print(x,y)
            '''diff = abs(width - w)
            if 0<diff<3:
                shape = "trapezium"
            else :
                shape = "rhombus"  '''
            #print(diff)
            cnt = approx
            
            #cv2.rectangle(image,(x,y),(x+w,y+h),(100,0,120),2)
            #ar = w / float(h)
            #print(ar)
            
            #pts = np.array(approx, dtype="int")
            #print("pts:",pts)

            # initialize a list of coordinates that will be ordered
            # such that the first entry in the list is the top-left,
            # the second entry is the top-right, the third is the
            # bottom-right, and the fourth is the bottom-left
            rect = np.zeros((4, 2), dtype="float32")
 
            # the top-left point will have the smallest sum, whereas
            # the bottom-right point will have the largest sum
            s = pts.sum(axis=1)
            rect[0] = pts[np.argmin(s)]
            rect[2] = pts[np.argmax(s)]
 
            # now, compute the difference between the points, the
            # top-right point will have the smallest difference,
            # whereas the bottom-left will have the largest difference
            diff = np.diff(pts, axis=1)
            rect[1] = pts[np.argmin(diff)]
            rect[3] = pts[np.argmax(diff)]
            
 
            # check to see if the new method should be used for
            # ordering the coordinates
	
            # show the re-ordered coordinates
            
            #print("DSDS")
            #print(rect.astype("int"))
            

            '''
            xSorted = pts[np.argsort(pts[:, 0]), :]
 
            # grab the left-most and right-most points from the sorted
            # x-roodinate points
            leftMost = xSorted[:2, :]
            rightMost = xSorted[2:, :]
 
            # now, sort the left-most coordinates according to their
            # y-coordinates so we can grab the top-left and bottom-left
            # points, respectively
            leftMost = leftMost[np.argsort(leftMost[:, 1]), :]
            (tl, bl) = leftMost
 
            # now that we have the top-left coordinate, use it as an
            # anchor to calculate the Euclidean distance between the
            # top-left and right-most points; by the Pythagorean
            # theorem, the point with the largest distance will be
            # our bottom-right point
            D = dist.cdist(tl[np.newaxis], rightMost, "euclidean")[0]
            (br, tr) = rightMost[np.argsort(D)[::-1], :]
 
            # return the coordinates in top-left, top-right,
            # bottom-right, and bottom-left order
            '''
            tl,tr,br,bl = np.array(rect, dtype="int")
            side1 = tl[0]-tr[0]
            side2 = bl[0] - br[0]
            difference = abs(side1 - side2) 
            if 0<=difference <3:
                shape = "rhombus"
            else:
                shape = "trapezium"
            #print(tl,tr,br,bl)



        
             
            # a square will have an aspect ratio that is approximately
            # equal to one, otherwise, the shape is a rectangle
            #shape = "square" if ar >= 0.95 and ar <= 1.05 else "rectangle"
 
            # if the shape is a pentagon, it will have 5 vertices
        elif len(approx) == 5:
           shape = "pentagon"
 
            # otherwise, we assume the shape is a circle

        elif len(approx) == 6:
           shape = "hexagon"

      
        else:
            shape = "circle"
 
        # return the name of the shape
        return shape



image = cv2.imread(args["image"])
gray = cv2.cvtColor(255 - image, cv2.COLOR_BGR2GRAY)
lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
#cv2.imshow("image",image)
#cv2.waitKey(0)

sd = ShapeDetector()
cl = ColorLabeler()

blurred = cv2.GaussianBlur(gray, (5, 5), 0)
thresh = cv2.threshold(blurred, 60, 255, cv2.THRESH_BINARY)[1]

# find contours in the thresholded image
cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
	cv2.CHAIN_APPROX_SIMPLE)
cnts = cnts[0] if imutils.is_cv2() else cnts[1]

# loop over the contours
for c in cnts:
        # compute the center of the contour
        M = cv2.moments(c)
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        # draw the contour and center of the shape on the image
        text = "{} \n {} \n {}".format(cl.label(lab, c), sd.detect(c),(cX,cY))
        y = [0,20,40]
        for i, line in enumerate(text.split('\n')):
                        
            cv2.putText(image,line, (cX - 20 + y[i], cY - 20 + y[i]),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)   
        (x, y, w, h) = cv2.boundingRect(c) 
        #print(x,y,w,h)
        print("aspect ratio:",float(w)/h) 

        cv2.drawContours(image, [c], -1, (255, 255, 0), 2)
        #cv2.circle(image, (cX, cY), 7, (255, 255, 255), -1)


 
# show the image
cv2.imshow("Image", image)
cv2.waitKey(0)

