#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import csv

with open('Data/categories.csv', mode='r') as infile:
    reader = csv.reader(infile)
    categories = {row[0]:i for i, row in enumerate(reader)}
    
with open('Data/train_labels.csv', mode='r') as infile:
    next(infile)
    reader = csv.reader(infile)
    labels = {int(row[0]):str(categories[row[1]]) for row in reader}    

# separate images by labels
images = np.load('Data/train_images.npy', encoding='bytes')
for i, image in enumerate(images):
    images[i][1] = image[1].reshape(100, 100) # previously 1, 100, 100, 1
    label = labels[images[i][0]]
    images[i][0] = label


# In[2]:


from sklearn.model_selection import train_test_split

X_train, X_val, y_train, y_val = train_test_split([item[1] for item in images], [item[0] for item in images], test_size=0.20, random_state=42)


# In[3]:


from skimage.measure import label,regionprops
from skimage.transform import resize

for i, img in enumerate(X_train):
    
    for k in range(0, len(img)):
            for j in range(0, len(img[k])):
                if img[k][j] < 50:
                    img[k][j] = 0
                else:
                    img[k][j] = 255
                    
    labelled_img = label(img, connectivity=2)
    regions = regionprops(labelled_img, img)

    biggest_area = 0;
    coords = []
    for region in regions:
        r1, c1, r2, c2 = region.bbox

        height = abs(c2 - c1)
        width = abs(r2 - r1)
        area = height * width
        if area > biggest_area:
            biggest_area = area
            coords = [r1, c1, r2, c2]
            
    height = abs(coords[3] - coords[1])
    width = abs(coords[2] - coords[0])
    area = height * width
    if area < 100:     #if bounding box area is less than the minimum area specified assume empty image
        X_train[i] = np.zeros((50,50))
    else:
        
        selected_img = img[min(coords[0],coords[2]):max(coords[0],coords[2]),min(coords[1],coords[3]):max(coords[1],coords[3])]

        X_train[i] = resize(selected_img,(50,50))
    
    
    
for i, img in enumerate(X_val):
    
    for k in range(0, len(img)):
            for j in range(0, len(img[k])):
                if img[k][j] < 50:
                    img[k][j] = 0
                else:
                    img[k][j] = 255
                    
    labelled_img = label(img, connectivity=2)
    regions = regionprops(labelled_img, img)

    biggest_area = 0;
    coords = []
    for region in regions:
        r1, c1, r2, c2 = region.bbox

        height = abs(c2 - c1)
        width = abs(r2 - r1)
        area = height * width
        if area > biggest_area:
            biggest_area = area
            coords = [r1, c1, r2, c2]
            
    height = abs(coords[3] - coords[1])
    width = abs(coords[2] - coords[0])
    area = height * width
    if area < 100:     #if bounding box area is less than the minimum area specified assume empty image
        X_val[i] = np.zeros((50,50))
    else:
        
        selected_img = img[min(coords[0],coords[2]):max(coords[0],coords[2]),min(coords[1],coords[3]):max(coords[1],coords[3])]

        X_val[i] = resize(selected_img,(50,50))


# In[4]:


print(len(X_train), len(X_val))


# In[ ]:


import matplotlib.pyplot as plt

plt.imshow(X_train[1], cmap=plt.cm.binary)
plt.show()


# In[5]:


# create directories to store processed images
import os
for directory in range(31):
    train = "data/train/"+str(directory)
    if not os.path.exists(train):
        os.makedirs(train)
    valid = "data/valid/"+str(directory)
    if not os.path.exists(valid):
        os.makedirs(valid)


# In[6]:


from PIL import Image

i = 0
for img_array, label in zip(X_train, y_train):
    im = Image.fromarray(img_array)
    im.save("data/train/"+label+"/"+str(i)+".tif")
    i+=1
    
i = 0
for img_array, label in zip(X_val, y_val):
    im = Image.fromarray(img_array)
    im.save("data/valid/"+label+"/"+str(i)+".tif")
    i+=1


# In[7]:


from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K


# dimensions of our images.
# EDIT THIS
img_width, img_height = 100, 100
train_data_dir = 'data/train'
validation_data_dir = 'data/valid'

nb_train_samples = 8000
nb_validation_samples = 2000
epochs = 50
batch_size = 16

if K.image_data_format() == 'channels_first':
    input_shape = (3, img_width, img_height)
else:
    input_shape = (img_width, img_height, 3)


# In[8]:


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

# model = Sequential()

# model.add(Flatten(input_shape=input_shape))
# model.add(Dense(128, activation='relu'))

# model.add(Dense(128, activation='relu'))

# model.add(Dense(31, activation='softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])


# In[9]:


# this is the augmentation configuration we will use for training
train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)


# In[10]:


# this is the augmentation configuration we will use for testing:
# only rescaling
test_datagen = ImageDataGenerator(rescale=1. / 255)

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical')

validation_generator = test_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical')

model.fit_generator(
    train_generator,
    steps_per_epoch=nb_train_samples // batch_size,
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=nb_validation_samples // batch_size)

model.save_weights('first_try.h5')

