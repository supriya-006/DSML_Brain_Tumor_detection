import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


print("Welcome to Brain Tumor Detector!")
print("Project is ready to start!")


import os
if os.path.exists('/home/supriya-devkota/Desktop/DSML/DSML_project/BrainTumorYolov8'):
    print("Data folder found!")
else:
    print("Please download data first!")