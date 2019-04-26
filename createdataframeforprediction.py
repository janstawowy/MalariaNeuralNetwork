import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cv2
import glob


W=32
H=32
images = [cv2.resize(cv2.imread(file),(W,H)) for file in glob.glob("photosforprediction/*.png")]

"""
#code for testing if images read succesfuly and if 
print(len(images))
print(len(labels))
print(labels[0])
print(images[0])
plt.imshow(images[0])
plt.show()"""

dict= {'image':images}
df=pd.DataFrame(data=dict)

df.to_pickle("./malariapredictdata.pkl")
