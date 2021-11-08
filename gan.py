import tensorflow as tf
import tensorflow.keras.layers as layers
import tensorflow.keras.losses as losses
import tensorflow.keras.optimizers as opt
import tensorflow.keras.models as models
import pandas as pd
from PIL import Image
import numpy as np 
import os
import matplotlib.pyplot as plt

def main():
    raw = getRawData()
    X_train, X_test, y_train, y_test = preprocessData(raw)
    model = trainModel((X_train,y_train))
    preds = getPredictions((X_test,y_test),model)

##### Model Pipeline #####
def getRawData():
    print("Loading Raw Images")
    train_images = get_training_images()
    test_images = get_test_images()
    return train_images,test_images

def preprocessData(raw):
    x,y = raw
    num_imgs = len(x)
    split = .8
    train_size = num_imgs * split
    test_size = num_imgs * (1-split)
    
    return (x_train,x_test,y_train,y_test)


def make_generator():
    pass

def make_discriminator():
    pass

def trainModel(data):
    x,y = data
    model = models.Sequential([
        layers.Dense(256,activation='relu'),
        layers.Dropout(.2),
        layers.Dense(1)
    ])
    loss = losses.MeanSquaredError()
    model.compile(optimizer='adam',loss=loss,metrics=['accuracy'])
    model.fit(x,y,epochs=5)
    return model

def getPredictions(data,model):
    x_test,y_test = data
    preds = model.predict(x_test)
    return preds

    
    
###### Helper Functions ######
def normalize(data):
    data = pd.DataFrame(data)
    for col in data.columns:
        data[col] = normalize_col(data[col])
        
    return data

def normalize_col(col):
    return (col-col.min())/(col.max()-col.min()) #max_min normalization
    return (col - col.mean())/(col.std()) # z-score normalization (better if outliers exist)

def get_training_images():
    print("\tLoading Training Images\t", end='')
    i = 0
    images = np.array([])
    for image in os.listdir('low_res'):
        if i % 100 == 0: 
            print(".",end='')
        np.append(images,np.array(Image.open("low_res/"+image)))
        i +=1
    print("\tDone")
    return images

def get_test_images():
    print("\tLoading Test Images\t", end='')
    i = 0
    images = np.array([])
    for image in os.listdir('high_res'):
        if i % 100 == 0: 
            print(".",end='')
        np.append(images,np.array(Image.open("high_res/"+image)))
        i +=1
        
    print("\tDone")
    return images

if __name__ == '__main__':
    main()