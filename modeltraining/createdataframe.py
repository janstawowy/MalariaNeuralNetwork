import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cv2
import glob


W=32
H=32
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

dict= {'image':images, 'state':labels}
df_infected=pd.DataFrame(data=dict)

images_uninfected = [cv2.resize(cv2.imread(file),(W,H)) for file in glob.glob("cell_images/Uninfected/*.png")]
labels_uninfected = ['uninfected' for i in range(len(images_uninfected))]

dict_uninf = {'image':images_uninfected, 'state':labels_uninfected}
df_uninfected=pd.DataFrame(data=dict_uninf)

malariadata = pd.concat([df_infected, df_uninfected], ignore_index=True)

"""
#check if df creation and merge were succesful
print(df.head())
print(df.tail())
"""

malariadata.to_pickle("./malariadata.pkl")
