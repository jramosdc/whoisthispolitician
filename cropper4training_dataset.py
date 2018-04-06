#http://answers.opencv.org/question/90010/opencv-python-face-crop-program/
#
#to reduce false positives, you can try to increase the 'minNeighbours' param in detectMultiScale()
#Ye thanks I found that 500 x 500 worked best with 1.7 and 5
#
#https://realpython.com/blog/python/face-recognition-with-python/
#If you want to use OpenCV3 just change "cv2.cv.CV_HAAR_SCALE_IMAGE" to "cv2.CASCADE_SCALE_IMAGE", it works for me
#faces = faceCascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(20, 20), flags = cv2.CASCADE_SCALE_IMAGE )

import numpy as np
import cv2
import os, os.path

#multiple cascades: https://github.com/Itseez/opencv/tree/master/data/haarcascades
#https://github.com/Itseez/opencv/blob/master/data/haarcascades/haarcascade_frontalface_default.xml
face_cascade = cv2.CascadeClassifier('faces.xml')
#https://github.com/Itseez/opencv/blob/master/data/haarcascades/haarcascade_eye.xml
eye_cascade = cv2.CascadeClassifier('eye.xml')
nombre='politiciansfolder'
DIR = 'staging/'+nombre
numPics = len([name for name in os.listdir(DIR) if os.path.isfile(os.path.join(DIR, name))])

nfaces_detected = 0

for pic in range(1, (numPics+1)):
    try:

    # note the dependency on the format of the filename 
        img = cv2.imread('staging/'+nombre+'/'+str(pic)+'.jpg')
        height = img.shape[0]
        width = img.shape[1]
        size = height * width
#???
#    if size > (500^2):
#        r = 500.0 / img.shape[1]
#        dim = (500, int(img.shape[0] * r))
#        img2 = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
#        img = img2

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=2, minSize=(15, 15), flags = cv2.CASCADE_SCALE_IMAGE )

        nface_within_pic = 0
    except:
        print ("Some images were not processed")
    for (x,y,w,h) in faces:
        face_with_eyes_detected = 0
        imgCrop = img[y:y+h,x:x+w]
        #cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]

        #eyes = eye_cascade.detectMultiScale(roi_gray)
        eyes = face_cascade.detectMultiScale(roi_gray, scaleFactor=1.3, minNeighbors=2, minSize=(5, 5), flags = cv2.CASCADE_SCALE_IMAGE )
        eyesn = 0
        for (ex,ey,ew,eh) in eyes:
            #cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
            eyesn = eyesn +1
# allow detection if only one 1 eye for sideways face profile ?  
# No, always assume a frontal profile since that's the haar detection profile we chose above
#        if eyesn >= 1:
        if eyesn >= 1:
            face_with_eyes_detected = 1
        #cv2.imshow('img',imgCrop)
        if face_with_eyes_detected > 0:
            cv2.imwrite("training_dataset/"+name+"/crop"+str(pic)+ "_" +str(nface_within_pic)+".jpg", imgCrop)
            print("Image"+str(pic)+ "_" +str(nface_within_pic)+" has been processed and cropped")
            nface_within_pic += 1
            nfaces_detected += 1
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

print("All "+str(numPics+1)+" images have been processed with " + str(nfaces_detected) + " faces detected and saved.")
print (numPics)

cv2.destroyAllWindows()