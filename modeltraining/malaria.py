import pandas as pd
import numpy as np
import keras
from keras.layers import Dense, Dropout, Conv2D, Flatten, MaxPool2D
from keras.models import Sequential
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from keras.layers.normalization import BatchNormalization
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

#load dataframe
data_path='malariadata.pkl'
df=pd.read_pickle(data_path)

"""
#check basic parameters of df and print example images with corresponding label(infected or not)
print(df.shape)
print(df['image'].iloc[0].shape)
print(df.head())
print(df.tail())
plt.figure(1)
plt.imshow(df['image'].iloc[0])
plt.title(df['state'].iloc[0])
plt.figure(2)
plt.imshow(df['image'].iloc[-1])
plt.title(df['state'].iloc[-1])
plt.show()
"""
#extract images and labels
target=df['state']
images=np.asarray(list(df['image']))

#hotencode labels
target[target == 'infected'] =1
target[target == 'uninfected'] = 0
target=np.asarray(list(target))

#split dataset into training and testing
X_train, X_test, y_train, y_test = train_test_split(images, target, test_size = 0.2, random_state=42)

#create model and add layers
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

#compile model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
#print model summary
model.summary()
#train model on train data
model_training=model.fit(X_train, y_train, validation_split = 0.2, epochs=50, batch_size=10)
#save model
model.save_weights("malariamodel.h5")

#plot loss and accuracy over epochs 
plt.figure(1)
plt.plot(model_training.history['val_loss'], 'r', label='validation')
plt.plot(model_training.history['loss'], 'b', label='train')
plt.title('loss')
plt.xlabel('Epochs')
plt.ylabel('Score')
plt.legend(loc='best', shadow=True)
plt.figure(2)
plt.plot(model_training.history['val_acc'], 'r', label='validation')
plt.plot(model_training.history['acc'], 'b', label='train')
plt.title('accuracy')
plt.xlabel('Epochs')
plt.ylabel('Score')
plt.legend(loc='best', shadow=True)
plt.show()

#predict on test dataset
y_pred=model.predict(X_test)
y_pred=(y_pred>0.5)

#print how well the model performs on unseen data
print('Confusion matrix:')
print(confusion_matrix(y_test, y_pred))
print('Classification report:')
print(classification_report(y_test, y_pred))
print('Accuracy on previously unseen data:')
print(accuracy_score(y_test, y_pred))
