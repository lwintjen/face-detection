"""
Author: Loris Wintjens
Goal : Face detection 
"""
import cv2
import numpy as np
from matplotlib import pyplot as plt
import glob

# read all the .jpg files in the working directory and put them into fileNames
fileNames = [] 
for file in glob.glob("*.jpg"):
    fileNames.append(file)
    
for imgFile in fileNames:
    #read each of the .jpg file contained in fileNames
    img = cv2.imread(imgFile,cv2.IMREAD_COLOR)
    imgCopy = img.copy() # we copy the img to not alter the img
    # as openCV face detector expects gray image, we convert the img
    imgCopyGray = cv2.cvtColor(imgCopy, cv2.COLOR_BGR2GRAY)

    facecascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

    eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
   
    faces = facecascade.detectMultiScale(imgCopyGray, scaleFactor=1.2, minNeighbors=5)

 
    print('Total number of Faces found',len(faces))
    
    # we loop through the list of faces and draw rectangles on the image
    for (x, y, w, h) in faces:
        faceDetect = cv2.rectangle(imgCopy, (x, y), (x+w, y+h), (255, 0, 255), 2)
        roi_gray = imgCopy[y:y+h, x:x+w]
        roi_color = imgCopy[y:y+h, x:x+w]
        plt.imshow(faceDetect)
        eyes = eye_cascade.detectMultiScale(roi_gray)
        # check for eyes detection
        for (ex,ey,ew,eh) in eyes:
            eye_detect = cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(255,0,255),2)
            plt.imshow(eye_detect)