from keras.layers import Dense, Dropout, Flatten, BatchNormalization
from keras.layers import Conv2D, MaxPooling2D, Activation
from sklearn.metrics import confusion_matrix
from keras.models import Sequential
from keras.models import load_model
import matplotlib.pyplot as plt
from keras import backend as K
import pandas as pd
import numpy as np
import itertools
import keras
import cv2

emotion = ['Angry', "Disgust", 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
img_rows, img_cols = 48, 48
num_classes = 7

data = pd.read_csv('data/fer2013.csv', delimiter=',')
data_test = data[32298:]
y_test = data_test['emotion'].values

x_test = np.zeros((y_test.shape[0], 48*48))
for i in range(y_test.shape[0]):
    x_test[i] = np.fromstring(data_test['pixels'][32298+i], dtype=int, sep=' ')

x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
x_test_rev = np.flip(x_test, 2)

input_shape = (img_rows, img_cols, 1)
x_test = x_test.astype('float32')
x_test_rev = x_test_rev.astype('float32')
x_test /= 255
x_test_rev /= 255
y_test = keras.utils.to_categorical(y_test, num_classes)

print('Loading Models...')
print('0/3')
model = []
for i in range(2):
    m = load_model('saved_model/' + 'cnn'+str(i)+'.h5')
    print(str(i+1) + '/3')
    model.append(m)

m = load_model('saved_model/ensemble.h5')
model.append(m)
print('3/3')
print("Loading Complete!")

def plot_confusion_matrix(cm):
    print(cm)
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix on Private Test Data')
    plt.colorbar()
    tick_marks = np.arange(len(emotion))
    plt.xticks(tick_marks, emotion)
    plt.yticks(tick_marks, emotion)

    fmt = '.2f'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True Emotions')
    plt.xlabel('Predicted Emotions')
    plt.tight_layout()

p = np.zeros((y_test.shape[0],14))
p[:,0:7] = model[0].predict(x_test)
p[:,7:14] = model[1].predict(x_test_rev)
y_pred = model[2].predict(p)
yp = np.argmax(y_pred, axis=1)
yt = np.argmax(y_test, axis=1)
cm = confusion_matrix(yt, yp)
plot_confusion_matrix(cm)
plt.savefig("img/cm.png")
plt.show()
