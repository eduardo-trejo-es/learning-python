# -*- coding: latin-1 -*-
"""
    Code 1.2 - Face detection in real time video
    This program detects faces with a webcam.
 
    Written by Glare and Transductor
    www.robologs.net
"""
import cv2
 
#We load our Haar classifier:
cascada_rostro = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')
# If you use another classifier or have it saved in a different directory than this python script,
# you will have to change 'haarcascade_frontalface_alt.xml' to the path to your .xml file.
 
#Start the webcam:
webcam = cv2.VideoCapture(0)
# NOTE 1: If it doesn't work, you can change the index 0 to another, or change it to the address of your webcam (eg '/ dev / video0')
# NOTE 2: it should also work if instead of a webcam you use a video file.

#We remind the user which is the key to exit:
print("\nReminder: press 'ESC' to close.\n")
 
 
while(1):
 
    #Capturing an image with the webcam:
    valid,img = webcam.read()
 
    #If the image is valid (that is, if it has been captured correctly), we continue:
    if valid:
 
        #Convert the image to gray:
        img_gris = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
 
 
        #We look for the faces:
        faces_coordinates = cascada_rostro.detectMultiScale(img_gris, 1.3, 5)
 
 
        #We go through the 'faces_coordinates' array and draw the rectangles on the original image:
        for (x,y,width, high) in faces_coordinates:
            cv2.rectangle(img, (x,y), (x+width, y+high), (0,0,255) , 3)
 
 
        #Abrimos una ventana con el resultado:
        cv2.imshow('Output', img)

        #Exit with 'ESC':
        k = cv2.waitKey(5) & 0xFF
        if k == 27:
            cv2.destroyAllWindows()
            break
 
webcam.release()
