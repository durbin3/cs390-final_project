import tensorflow as tf
import tensorflow.keras.layers as layers
import tensorflow.keras.losses as losses
import tensorflow.keras.optimizers as opt
import tensorflow.keras.models as models
import pandas as pd
import numpy as np 


def main():
    raw = getRawData('AAPL',2019)
    X_train, X_test, y_train, y_test = preprocessData(raw)
    model = trainModel((X_train,y_train))
    preds = getPredictions((X_test,y_test),model)

##### Model Pipeline #####
def getRawData():
    pass

def preprocessData(raw):
    X_train,X_test,y_train,y_test = {}
    return (X_train,X_test,y_train,y_test)

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



if __name__ == '__main__':
    main()