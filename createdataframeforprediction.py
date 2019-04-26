import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cv2
import glob

#Set width and height of images
W=32
H=32
#read and resize all images from a folder
images = [cv2.resize(cv2.imread(file),(W,H)) for file in glob.glob("photosforprediction/*.png")]

"""
#code for testing if images read succesfuly and if 
print(len(images))
print(len(labels))
print(labels[0])
print(images[0])
plt.imshow(images[0])
plt.show()"""

#create and save dataframe
dict= {'image':images}
df=pd.DataFrame(data=dict)

df.to_pickle("./malariapredictdata.pkl")
