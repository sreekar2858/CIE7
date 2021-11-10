import os

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten

gyrFile = 'Gyroscope'
accFile = 'Accelerometer'

rootdir = os.getcwd()
os.chdir(rootdir)

normal = 'normal'
impaired = 'impaired'
upstairs = 'upstairs'
downstairs = 'downstairs'

s1 = 'Smartphone1'
s2 = 'Smartphone2'
s3 = 'Smartphone3'
s4 = 'Smartphone4'

for subdir, dirs, files in os.walk(rootdir):
    if s1 in subdir:
        if normal in subdir:
            for file in files:
                if gyrFile in file:
                    print(os.path.join(subdir, file))
                    with open(os.path.join(subdir, file), 'r') as f: # open in readonly mode
                        gyrData = pd.read_csv(f)
                        gyrTime = gyrData['Time (s)']
                        gyrShape = gyrData.shape
                    
                        freqGyr = 1/(gyrTime[1]-gyrTime[0])
                    
                    print("Frequency of Gyroscope: ", freqGyr)
                    print(gyrShape)
                    
                    # print(gyrData.tail(1))
                    # print(accData.tail(1))


#End of file data import
