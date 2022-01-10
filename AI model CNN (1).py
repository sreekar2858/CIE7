#!/usr/bin/env python
# coding: utf-8

# In[16]:


import pickle
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation, Dense, Flatten, Dropout, Conv1D, Conv2D, MaxPooling1D, MaxPooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import categorical_crossentropy
import matplotlib.pyplot as plt

with open('train_resample.pickle', 'rb') as f:
    train_resample = pickle.load(f)
with open('train_norm_labels.pickle', 'rb') as f:
    train_norm_labels = pickle.load(f)

x = np.asarray(train_resample)
y = np.asarray(train_norm_labels)

size = 0.2

x_train, x_test, y_train, y_test = train_test_split(x,y,shuffle=True, test_size=size)

print(f'Shape of training data - {x_train.shape} \tShape of its labels - {y_train.shape} \nShape of testing data - {x_test.shape} \tShape of its labels') 
# model = Sequential()
# model.add(Conv2D(32, (3, 3), activation='relu', padding='same', name='conv_1', 
#                   input_shape=(740,100,4)))
# model.add(MaxPooling2D((2, 2), name='maxpool_1'))
# model.add(Conv2D(64, (3, 3), activation='relu', padding='same', name='conv_2'))
# model.add(MaxPooling2D((2, 2), name='maxpool_2'))
# model.add(Conv2D(128, (3, 3), activation='relu', padding='same', name='conv_3'))
# model.add(MaxPooling2D((2, 2), name='maxpool_3'))
# model.add(Conv2D(128, (3, 3), activation='relu', padding='same', name='conv_4'))
# model.add(MaxPooling2D((2, 2), name='maxpool_4'))
# model.add(Flatten())
# model.add(Dropout(0.5))
# model.add(Dense(512, activation='relu', name='dense_1'))
# model.add(Dense(128, activation='relu', name='dense_2'))
# model.add(Dense(2, activation='sigmoid', name='output'))

      # fit and evaluate a model
# def evaluate_model(trainX, trainy, testX, testy):
# 	verbose, epochs, batch_size = 0, 10, 32
# 	n_timesteps, n_features, n_outputs = trainX.shape[1], trainX.shape[2], trainy.shape[1]
model = Sequential()
model.add(Conv1D(filters=32, kernel_size=3, activation='relu', input_shape=(100,4)))
model.add(MaxPooling1D(pool_size=2))
model.add(Conv1D(filters=64, kernel_size=3, activation='relu'))
model.add(MaxPooling1D(pool_size=2))
model.add(Conv1D(filters=128, kernel_size=3, activation='relu'))
model.add(MaxPooling1D(pool_size=2))
# 	model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(2, activation='relu'))
model.summary()
# 	model.add(Dense(n_outputs, activation='softmax'))

# 	# fit network
# 	
# 	# evaluate model
# 	_, accuracy = model.evaluate(testX, testy, batch_size=batch_size, verbose=0)
# 	return accuracy
      


# In[17]:


model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


# In[18]:


history=model.fit(x_train, y_train, epochs=150, validation_data=(x_test,y_test), verbose=2, batch_size=32)


# In[ ]:




