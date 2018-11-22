#!/usr/bin/env python
# coding: utf-8

# In[18]:


# import labels and images from file

import numpy as np

# separate images by labels
images = np.load('Data/processed_test_images.npy', encoding='bytes')


# In[19]:


from keras.utils import np_utils

X = [item[1] for item in images]
y = [item[0] for item in images]

X_array = np.array(X)
X_array = X_array.reshape(X_array.shape[0], 50, 50, 1)
X_array = X_array.astype('float64')
X_array /= 255


# In[20]:


from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K


# dimensions of our images.
# EDIT THIS
img_width, img_height = 50,50

nb_train_samples = 6000
nb_validation_samples = 2000
epochs = 3
batch_size = 16

if K.image_data_format() == 'channels_first':
    input_shape = (1, img_width, img_height)
else:
    input_shape = (img_width, img_height, 1)


# In[21]:


model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=input_shape))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(31))
model.add(Activation('sigmoid'))

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])


# In[22]:


model.load_weights('weights/CNN-augmentation1.h5')


# In[24]:


predicted_classes = model.predict_classes(X_array)


# In[25]:


predicted_classes[:10]


# In[26]:


# open number to category csv file
import csv
with open('Data/categories.csv', mode='r') as infile:
    reader = csv.reader(infile)
    categories = {i:row[0] for i, row in enumerate(reader)}


# In[30]:


with open('submission.csv','w') as file:
    file.write('Id,Category\n')
    for i, prediction in enumerate(predicted_classes):
        file.write(str(i) + ',' + categories[prediction])
        file.write('\n')


# In[ ]:




