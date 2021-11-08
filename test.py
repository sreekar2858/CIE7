import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# from tensorflow import keras

gyrFile = "/home/sree/ExCIE/CIE_Project/SensorData/NormalWalking_60sec/Gyroscope.csv"
accFile = "/home/sree/ExCIE/CIE_Project/SensorData/NormalWalking_60sec/Accelerometer.csv"

gyrData = pd.read_csv(gyrFile)
gyrTime = gyrData['Time (s)']
gyrShape = gyrData.shape

accData = pd.read_csv(accFile)
accTime = accData['Time (s)']
accShape = accData.shape

freqGyr = 1/(gyrTime[1]-gyrTime[0])
freqAcc = 1/(accTime[1]-accTime[0])

print("Frequency of Gyroscope: ", freqGyr)
print("Frequency of Accelerometer: ", freqAcc)
print(gyrShape)
print(accShape)

print(gyrData.tail(1))
print(accData.tail(1))