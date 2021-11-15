import os

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten

# import data from other python file
import dataProc


# #AI model (sample)

targetLabels = tf.to_categorical(dataProc.gyrData[''])
xData = dataProc.gyrData.shape[1]

gyr_input_data = dataProc.gyrData.values

model = Sequential() 
model.add(Dense(units=128, input_shape = (xData,)))
model.add(Activation('relu'))
model.add(Dense(units=1, input_shape = (xData,)))
model.add(Activation('softmax'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(gyr_input_data, y_train, batch_size=32, epochs=10)

score = model.evaluate(gyr_input_data, y_test, batch_size=32)

model.predict(gyr_input_data)

print("\nLoss: ", score[0])
print("Accuracy: ", score[1])
