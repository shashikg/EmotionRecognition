import numpy as np
import cv2
from detectfaces import get_faces
from keras.models import load_model
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, BatchNormalization
from keras.layers import Conv2D, MaxPooling2D, Activation
from keras import backend as K
import matplotlib.pyplot as plt

img_rows, img_cols = 48, 48
# 0=Angry, 1=Disgust, 2=Fear, 3=Happy, 4=Sad, 5=Surprise, 6=Neutral
emotion = ['Angry', "Disgust", 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

model = []
for i in range(2):
    m = load_model('saved_model/' + 'cnn'+str(i)+'.h5')
    print('cnn '+str(i))
    model.append(m)
m = load_model('saved_model/ensemble.h5')
model.append(m)

print("Models Loaded...")

def predict(x):
    x_rev = np.flip(x, 1)
    p = np.zeros((1, 14))
    p[:,0:7] = model[0].predict(x.reshape(1,48,48,1))
    p[:,7:14] = model[1].predict(x_rev.reshape(1,48,48,1))
    pre = model[2].predict(p)

    return pre

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    cap.open()

while(True):
    ret, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = get_faces(gray)
    for i, (x,y,face) in enumerate(faces):
        cv2.imshow(str(i),face)
        pre = predict(face)
        # print(pre)
        print(emotion[np.argmax(pre)])

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
