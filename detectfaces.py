# Function to detect faces in a grascale image returns its top right corner and resized image of 48x48
# Author: Shashi Kant
# Date: 07/12/2018

import numpy as np
import cv2

face_cascade = cv2.CascadeClassifier('saved_model/haarcascade_frontalface_default.xml')

def get_faces(gray):
    faces = []
    faces_detected = face_cascade.detectMultiScale(gray, 1.3, 5)
    for i, (x,y,w,h) in enumerate(faces_detected):
        my = int(y + h/2)
        mx = int(x + w/2)

        if h<w:
            c = int(h/2)
        else:
            c = int(w/2)

        face = gray[my-c:my+c, mx-c:mx+c]
        face_48 = cv2.resize(face,(48, 48), interpolation = cv2.INTER_CUBIC)
        faces.append((y, x + w, face_48))

    return faces
