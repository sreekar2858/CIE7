#!/usr/bin/env python
# coding: utf-8

# In[4]:


import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation, Dense, Flatten, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import categorical_crossentropy
import matplotlib.pyplot as plt
import numpy as np



# In[42]:


phone=pd.read_csv('Trial_data.xls')


# In[43]:


phone


# In[162]:


dataset=phone.values


# In[33]:


dataset


# In[164]:


x=dataset[:,0:6]
y=dataset[:,6]



# In[165]:


min_max_scaler=preprocessing.MinMaxScaler()
x_scale=min_max_scaler.fit_transform(x) 
x_scale


# In[166]:


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, shuffle = True)
x_train=np.asarray(x_train).astype(np.float32)
x_test=np.asarray(x_test).astype(np.float32)


# In[167]:


print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)


# In[168]:


model = Sequential([
    Dense(32, activation='relu', input_shape=(6,)),
    Dense(32, activation='relu'),
    Dense(1, activation='softmax'),
])
# model.add(Flatten())


# In[169]:


model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy']) 


# In[170]:


#  loop for 2,5,10 folds
# iteration=0
# while(iteration<3):
# if(iteration==0)
#  n_splits=2
#     elif(iteration==1)
#     n_splits=5
#     else:
#         n_splits=10
# kfold = StratifiedKFold(n_splits, shuffle=True, random_state=0)
# cvscores = []
# for train, test in kfold.split(x, y):
#     model.add(Dense(12, input_dim=8, activation='relu'))
#     model.add(Dense(8, activation='relu'))
#     model.add(Dense(1, activation='softmax'))
#     model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
#     model.fit(x_train, y_train, epochs=150, batch_size=6, verbose=0)
#     model.add(Flatten())
#     scores = model.evaluate(x_test, y_test, verbose=0)
#     print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
#     cvscores.append(scores[1] * 100)
#     print("%.2f%% (+/- %.2f%%)" % (np.mean(cvscores), np.std(cvscores)))
#     history=model.fit(x_train, y_train,validation_data=(x_test, y_test), epochs=10, batch_size=6, verbose=1)
#     hist.append(history)
#     scores = model.evaluate(x_test, y_test, verbose=0)
#     model.output_shape
hist = model.fit(x_train, y_train,
        batch_size=6, epochs=10,
        validation_data=(x_test, y_test))
history.append(hist)
scores = model.evaluate(x_test, y_test, verbose=0)


# In[55]:


plt.plot(hist.history['loss'])
plt.plot(hist.history['train_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper right')
plt.show()


# In[ ]:


plt.plot(hist.history['accuracy'])
plt.plot(hist.history['Test_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='lower right')
plt.show()
# visualising training and testing accuracy

