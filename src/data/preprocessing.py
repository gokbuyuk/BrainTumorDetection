import os
import sys
sys.path.append(os.getcwd())

import cv2
import numpy as np
from collections import Counter
from icecream import ic
import matplotlib.pyplot as plt
from libs.preprocessor import Preprocessor
from libs.utils import *

import pandas as pd
from sklearn.model_selection import train_test_split

print("Preprocessing is running")
INPUT_PATH = 'data/raw' # Input path
OUTPUT_PATH = 'data/ml_ready'

# create output folder if it doesn't exist
if not os.path.exists(OUTPUT_PATH):
    os.makedirs(OUTPUT_PATH)
if not os.path.exists('reports'):
    os.makedirs('reports')

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


### for binary classification
target_label = 'binary'
output_subdir = 'binary'
# create this folder in OUTPUT_PATH if it doesn't exist
if not os.path.exists(os.path.join(OUTPUT_PATH, output_subdir)):
    os.makedirs(os.path.join(OUTPUT_PATH, output_subdir))
    
df_train["Binary_label"] = np.where(df_train["Label"]=="notumor",0,1)
X_train= df_train["Image_array"].to_numpy()
y_train= df_train["Binary_label"].to_numpy()
ic("Before split",y_train.mean().round(3))

df_test["Binary_label"] = np.where(df_test["Label"]=="notumor",0,1)
X_test= df_test["Image_array"].to_numpy()
y_test= df_test["Binary_label"].to_numpy()
ic("Before split",y_test.mean().round(3))

n_obs = X_train.shape[0]
ic(n_obs)

X_train = np.stack(X_train)

#scaling our train data
X_train = X_train/255
X_test = X_test/255

X_train_train, X_val, y_train_train, y_val = train_test_split(X_train, y_train,
                                                              test_size=.2,
                                                              shuffle=True,
                                                              stratify=y_train,
                                                              random_state=440)
ic(X_train_train.shape,X_val.shape,y_train_train.shape,y_val.shape, X_test.shape, y_test.shape)
ic("After split", y_train_train.mean().round(3),y_val.mean().round(3), y_test.mean().round(3))

for data, data_label in zip([X_train_train, X_val, y_train_train, y_val, X_test, y_test], 
                            ['X_train_train', 'X_val', 'y_train_train', 'y_val', 'X_test', 'y_test']):
    output_file_path = os.path.join(OUTPUT_PATH, output_subdir, f"{data_label}_{target_label}.npy")
    np.save(output_file_path, data)

## for Multi classification
target_label = 'multiclass'
output_subdir = 'multiclass'
# create this folder in OUTPUT_PATH if it doesn't exist
if not os.path.exists(os.path.join(OUTPUT_PATH, output_subdir)):
    dir_path = os.path.join(OUTPUT_PATH, output_subdir)
    os.makedirs(dir_path)
    print("Created folder: ", dir_path)

class_map = {'notumor': 0, 
             'meningioma': 1, 
             'pituitary': 2, 
             'glioma': 3}

df_train["multi_label"] = df_train["Label"].map(class_map)
X_train= df_train["Image_array"].to_numpy()
y_train= df_train["multi_label"].to_numpy()
ic("Before split",y_train.mean())

df_test["multi_label"] = df_test["Label"].map(class_map)
ic(df_test.head())
ic(df_test["multi_label"].value_counts(normalize=True))
X_test= df_test["Image_array"].to_numpy()
y_test= df_test["multi_label"].to_numpy()
ic(y_test)


n_obs = X_train.shape[0]
ic(n_obs)

X_train = np.stack(X_train)

#scaling our train data
X_train = X_train/255
X_test = X_test/255

X_train_train, X_val, y_train_train, y_val = train_test_split(X_train, y_train,
                                                              test_size=.2,
                                                              shuffle=True,
                                                              stratify=y_train,
                                                              random_state=440)
ic(X_train_train.shape,X_val.shape,y_train_train.shape,y_val.shape, X_test.shape, y_test.shape)
ic("After split", Counter(y_train_train),
   Counter(y_val), 
   Counter(y_test))

for data, data_label in zip([X_train_train, X_val, y_train_train, y_val, X_test, y_test], 
                            ['X_train_train', 'X_val', 'y_train_train', 'y_val', 'X_test', 'y_test']):
    # now save the output to the folder
    output_file_path = os.path.join(OUTPUT_PATH, output_subdir, f"{data_label}_{target_label}.npy")
    np.save(output_file_path, data)
