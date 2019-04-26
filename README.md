# MalariaNeuralNetwork
A simple convolutional neural networks that can be used to distinguish photos of cells that are infected with malaria from those that are not.

## Features:
Predict if the cell is infected with malaria basing on it's microscope photo
Create pandas dataframe with photos
Train model with photos

## How to use:
To predict if cell is infected simpy put microscope photo (or photos) in photosforprediction folder, create dataframe by running createdataframeforprediction.py and then run predict.py. Result should be an image of the cell titled True if the cell is infected or False if it is not. Model should be around 95% correct on unseen data.

To train model put a couple thousands of microscope images of infected cells in model_training/cell_images/Parasitized folder and a couple thousands ofmicroscope images of uninfected cells in model_training/cell_images/Uninfected.

Photos can be found on https://ceb.nlm.nih.gov/repositories/malaria-datasets/

## Tools
Python, Keras, matplotlib.pyplot, Pandas, NumPy, OpenCV, Sklearn

## Screenshots
![malariascores](https://user-images.githubusercontent.com/40367586/56801154-12303400-681d-11e9-9bc2-f45482a05088.png)

![accumalaria](https://user-images.githubusercontent.com/40367586/56801168-18261500-681d-11e9-89b0-11de6f76e445.png)

![lossmalaria](https://user-images.githubusercontent.com/40367586/56801170-19574200-681d-11e9-9aae-ecc39f29760f.png)
