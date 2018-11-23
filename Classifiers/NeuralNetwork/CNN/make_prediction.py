#!/usr/bin/env python
# coding: utf-8

# In[18]:


# import labels and images from file

import numpy as np
import sys
sys.path.append("../../../")
from dataProcessor import ImageFileHandler
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K

# Variables
test_src = '../../../Data/Processed/train_data_non_bin.npy'
categories_csv = '../../../Data/categories.csv'
weight_file = 'weights/save.h5'

img_width, img_height = 50,50

output_csv = 'submission.csv'

# separate images by labels
images = ImageFileHandler(test_src)


# In[19]:

X = images.xMatrix

X_array = np.array(X)
X_array = X_array.reshape(X_array.shape[0], img_width, img_height, 1)
X_array = X_array.astype('float64')
X_array /= 255


# In[20]:



# dimensions of our images

# nb_train_samples = 6000
# nb_validation_samples = 2000
input_shape = (img_width, img_height, 1)

# In[21]:


model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=input_shape))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(128, (3, 3)))
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


model.load_weights(weight_file)


# In[24]:


predicted_classes = model.predict_classes(X_array)


# In[25]:


predicted_classes[:10]


# In[26]:


# open number to category csv file
import csv
with open(categories_csv, mode='r') as infile:
    reader = csv.reader(infile)
    categories = {i:row[0] for i, row in enumerate(reader)}


# In[30]:


with open(output_csv,'w') as file:
    file.write('Id,Category\n')
    for i, prediction in enumerate(predicted_classes):
        file.write(str(i) + ',' + categories[prediction])
        file.write('\n')


# In[ ]:




