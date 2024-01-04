import numpy as np
import pandas as pd 
from icecream import ic

from keras import models, layers, optimizers, losses, metrics
from keras.utils import to_categorical


for data_label in ['X_train_train', 'X_val', 'y_train_train', 'y_val']:
    df=np.load(f"data/ml_ready/{data_label}.npy")
    ic(df.shape)
    globals()[data_label] = df

#make an empty model object
model = models.Sequential()

#adding convolutional and pooling layers

model.add( layers.Conv2D(32, (3,3),activation="relu", input_shape= (120,120,3)))

model.add( layers.MaxPooling2D((2,2),strides=2))

#adding another layer

model.add( layers.Conv2D(64, (3,3),activation="relu", input_shape= (120,120,3)))
model.add( layers.MaxPooling2D((2,2),strides=2))

#flatting
model.add(layers.Flatten())

model.add(layers.Dense(64, activation= "relu"))

#classification, output layer
model.add(layers.Dense(2, activation= "sigmoid"))

ic(model.summary())

