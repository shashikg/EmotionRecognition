import numpy as np
import cv2

face_cascade = cv2.CascadeClassifier('saved_model/haarcascade_frontalface_default.xml')

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    cap.open()

while(True):
    ret, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for i, (x,y,w,h) in enumerate(faces):
        img = cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]
        my = y + h/2
        mx = x + w/2
        if h<w:
            c = h/2
        else:
            c = w/2

        face = img[my-c:my+c, mx-c:mx+c]
        face_48 = cv2.resize(face,(48, 48), interpolation = cv2.INTER_CUBIC)
        cv2.imshow(str(i),face_48)

    cv2.imshow('img',img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
out.release()
cv2.destroyAllWindows()
