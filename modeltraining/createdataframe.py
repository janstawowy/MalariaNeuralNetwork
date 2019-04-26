import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cv2
import glob

#set width and height of images
W=32
H=32
#read and resize all images of infected cellsfrom a folder and label them as infected
images = [cv2.resize(cv2.imread(file),(W,H)) for file in glob.glob("cell_images/Parasitized/*.png")]
labels = ['infected' for i in range(len(images))]
"""
#code for testing if images read succesfuly and if 
print(len(images))
print(len(labels))
print(labels[0])
print(images[0])
plt.imshow(images[0])
plt.show()"""

#create dataframe with infected cells and labels
dict= {'image':images, 'state':labels}
df_infected=pd.DataFrame(data=dict)

#read and resize all images of uninfected cellsfrom a folder and label them as uninfected
images_uninfected = [cv2.resize(cv2.imread(file),(W,H)) for file in glob.glob("cell_images/Uninfected/*.png")]
labels_uninfected = ['uninfected' for i in range(len(images_uninfected))]

#create dataframe with uninfected cells
dict_uninf = {'image':images_uninfected, 'state':labels_uninfected}
df_uninfected=pd.DataFrame(data=dict_uninf)

#merge infected and uninfected dataframes
malariadata = pd.concat([df_infected, df_uninfected], ignore_index=True)

"""
#check if df creation and merge were succesful
print(df.head())
print(df.tail())
"""
#save dataframe 
malariadata.to_pickle("./malariadata.pkl")
