import pandas as pd
import numpy as np
import keras
from keras.layers import Dense, Dropout, Conv2D, Flatten, MaxPool2D
from keras.models import Sequential
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from keras.layers.normalization import BatchNormalization
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

data_path='malariapredictdata.pkl'
df=pd.read_pickle(data_path)

#extract images and labels
images=np.asarray(list(df['image']))


model=Sequential()
model.add(Conv2D(32, kernel_size=(3,3),padding='same', activation='relu', input_shape=(32, 32, 3,)))
model.add(MaxPool2D(2))
model.add(BatchNormalization())
model.add(Conv2D(32, kernel_size=(3,3),padding='same', activation='relu'))
model.add(MaxPool2D(2))
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.load_weights('malariamodel.h5')


y_pred=model.predict(images)
y_pred=(y_pred>0.5)
print(y_pred)

i=0
for image in images:
    plt.figure(i+1)
    plt.imshow(image)
    plt.title(y_pred[i])
    i=i+1

plt.show()