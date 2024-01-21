import os
import sys
sys.path.append(os.getcwd())

import cv2
import numpy as np
from icecream import ic
import matplotlib.pyplot as plt
from libs.preprocessor import Preprocessor
from libs.utils import *

import pandas as pd
from sklearn.model_selection import train_test_split

INPUT_PATH = 'data/raw' # Input path
OUTPUT_PATH = 'data/ml_ready'
# create output folder if it doesn't exist
if not os.path.exists(OUTPUT_PATH):
    os.makedirs(OUTPUT_PATH)

# Preprocess the training and testing data
preprocessor = Preprocessor()
preprocessor.process_directory(INPUT_PATH, INPUT_PATH.replace('raw', 'interim/resized'))


# Create DataFrames with metadata of the preprocessed data
train_df = get_metadata('data/interim/resized/Training', 'train')
test_df = get_metadata('data/interim/resized/Testing', 'test')
# Combine the DataFrames
df_preprocessed = pd.concat([train_df, test_df], ignore_index=True)
ic(df_preprocessed.head()) # print the df head
# Save the DataFrame to a CSV file
df_preprocessed.to_csv('reports/image_sizes_labels_and_data_preprocessed.csv', index=False)

# Get the image array for each image and add it to the data frame with metadata
df_preprocessed['Image_array'] = df_preprocessed['Path'].apply(lambda x: preprocessor.get_image_array(x))
df_train = df_preprocessed[df_preprocessed['Data']=='train']
df_train.to_csv('data/processed/train_data_preprocessed.csv', index=False)
df_test= df_preprocessed[df_preprocessed['Data']=='test']
df_test.to_csv('data/processed/test_data_preprocessed.csv', index=False)



df_train["Binary_label"] = np.where(df_train["Label"]=="notumor",0,1)
X_train= df_train["Image_array"].to_numpy()
y_train= df_train["Binary_label"].to_numpy()
ic("Before split",y_train.mean().round(3))

n_obs = X_train.shape[0]
ic(n_obs)

X_train = np.stack(X_train)

#scaling our train data
X_train = X_train/255

X_train_train, X_val, y_train_train, y_val = train_test_split(X_train, y_train,
                                                              test_size=.2,
                                                              shuffle=True,
                                                              stratify=y_train,
                                                              random_state=440)
ic(X_train_train.shape,X_val.shape,y_train_train.shape,y_val.shape)
ic("After split", y_train_train.mean().round(3),y_val.mean().round(3))


for data, data_label in zip([X_train_train, X_val, y_train_train, y_val], ['X_train_train', 'X_val', 'y_train_train', 'y_val']):
    np.save(f"data/ml_ready/{data_label}.npy", data)

