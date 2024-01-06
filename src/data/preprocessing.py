import os
import sys
sys.path.append(os.getcwd())

import cv2
import numpy as np
from icecream import ic
import matplotlib.pyplot as plt
from libs.preprocessor import Preprocessor
from libs.utils import *


INPUT_PATH = 'data/raw' # Input path
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

# Save the preprocessed training and testing data separately 
df_preprocessed[df_preprocessed['Data']=='train'].to_csv('data/processed/train_data_preprocessed.csv', index=False)
df_preprocessed[df_preprocessed['Data']=='test'].to_csv('data/processed/test_data_preprocessed.csv', index=False)
