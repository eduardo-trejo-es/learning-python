#classifiers in .xml format
#https://github.com/opencv/opencv/tree/master/data/haarcascades 

# -*- coding: latin-1 -*-
"""
    Code 1.1 - Face Detection in a Still Image
    This program finds all the faces in a photo.
 
    Written by Glare and Transductor
    www.robologs.net
"""
import cv2
 
#We load our Haar classifier:
waterfall_face = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')
# If you use another classifier or have it saved in a different directory than this python script,
# you will have to change 'haarcascade_frontalface_alt.xml' to the path to your xml file.
 
 
#We load the image and convert it to gray:test_image_CV.jpg
img = cv2.imread('test_image_CV.jpg')
img_gris = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# Note: the example image that we used for the tutorial is already in black and white,
# so it would not be necessary to convert it. I've done it anyway in case you want to later
# try a color image.
 
 
#We look for the faces:
face_coordinates = waterfall_face.detectMultiScale(img_gris, 1.3, 5)
# Note 1: the detectMultiScale () function requires a grayscale image. This is the reason
# why we have converted from BGR to Grayscale.
# Note 2: '1.3' and '5' are standard parameters for this function. The first is the scale factor ('scaleFactor'): the
# function will try to find faces by scaling the image multiple times, and this factor indicates by how much the image is reduced
# every time. The second parameter is called 'minNeighbours' and indicates the quality of the detections: a high value
# results in fewer detections but more reliability.
 
 
#Now we go through the 'face_coordinates' array and draw the rectangles over the original image:
for (x,y,width, high) in face_coordinates:
    cv2.rectangle(img, (x,y), (x+width, y+high), (0,0,255) , 3)
 
 
#We open a window with the result:
cv2.imshow('Output', img)
print("\nShowing result. Press any key to exit.\n")
cv2.waitKey(0)
cv2.destroyAllWindows()
