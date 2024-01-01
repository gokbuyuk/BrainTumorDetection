import os
import sys
sys.path.append(os.getcwd())

import cv2
import numpy as np
from icecream import ic
import matplotlib.pyplot as plt
from libs.preprocessor import Preprocessor


INPUT_PATH = 'data/raw' # Input path
preprocessor = Preprocessor()
preprocessor.process_directory(INPUT_PATH, INPUT_PATH.replace('raw', 'interim/resized'))

