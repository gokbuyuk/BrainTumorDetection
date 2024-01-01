import os

print('---- Working directory is:')
print(os.getcwd())

import sys
# sys.path.append("../libs")

import cv2
import numpy as np
from icecream import ic
import matplotlib.pyplot as plt
# TODO: fix import from libs.utils import *
from libs.utils import *

# Input paths
INPUT_PATH = 'data/raw'

crop_black_frame_in_directory(INPUT_PATH, INPUT_PATH.replace('raw', 'interim/cropped'))
downsample_images('data/interim/cropped', 'data/interim/resized')
