#!/usr/bin/env python
# coding: utf-8

# In[1]:


# import labels and images from file

import numpy as np

# separate images by labels
images = np.load('Data/processed_train_images.npy', encoding='bytes')


# In[2]:


from keras.utils import np_utils

X = [item[1] for item in images]
y = [item[0] for item in images]

X_array = np.array(X)
X_array = X_array.reshape(X_array.shape[0], 50, 50, 1)
X_array = X_array.astype('float64')
X_array /= 255

y_array = np.array(y)
y_array = np_utils.to_categorical(y_array, 31)


# In[3]:


y_array[0]


# In[4]:


# train-validation split
from sklearn.model_selection import train_test_split

X_other, X_val, y_other, y_val = train_test_split(X_array, y_array, test_size=0.20, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X_other, y_other, test_size=0.25, random_state=24)


# In[5]:


print(len(X_train), len(X_val), len(X_test))


# In[8]:


import matplotlib.pyplot as plt
for i in range(9):
    plt.subplot(3,3,i+1)
    plt.imshow(X_train[i].reshape(50,50), cmap='gray', interpolation='none')
    plt.title("Class {}".format(np.argmax(y_train[i])))


# In[9]:


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


# In[10]:


model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=input_shape))
model.add(Activation('relu'))
# model.add(MaxPooling2D(pool_size=(2, 2)))

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


# In[56]:


# model.load_weights('weights/CNN-augmentation1.h5')


# In[11]:


# this is the augmentation configuration we will use for training
train_datagen = ImageDataGenerator(
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)


# In[ ]:


# this is the augmentation configuration we will use for testing:
# only rescaling
test_datagen = ImageDataGenerator()

train_generator = train_datagen.flow(
    X_train,
    y_train,
    batch_size=batch_size)

validation_generator = test_datagen.flow(
    X_val,
    y_val,
    batch_size=batch_size)

model.fit_generator(
    train_generator,
    steps_per_epoch=nb_train_samples // batch_size,
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=nb_validation_samples // batch_size)

model.save_weights('weights/CNN-augmentation1.h5')


# In[57]:


score = model.evaluate(X_test, y_test, verbose=0)
print('Test score:', score[0])
print('Test accuracy:', score[1])


# In[47]:


# predicted_classes = model.predict_classes(X_test)
# correct_indices = np.nonzero(predicted_classes == [np.argmax(i) for i in y_test])[0]
# incorrect_indices = np.nonzero(predicted_classes != [np.argmax(i) for i in y_test])[0]


# In[58]:


# plt.figure()
# for i, correct in enumerate(correct_indices[:9]):
#     plt.subplot(3,3,i+1)
#     plt.imshow(X_test[correct].reshape(50,50), cmap='gray', interpolation='none')
#     plt.title("Predicted {}, Class {}".format(predicted_classes[correct], np.argmax(y_test[correct])))
    
# plt.figure()
# for i, incorrect in enumerate(incorrect_indices[:9]):
#     plt.subplot(3,3,i+1)
#     plt.imshow(X_test[incorrect].reshape(50,50), cmap='gray', interpolation='none')
#     plt.title("Predicted {}, Class {}".format(predicted_classes[incorrect], np.argmax(y_test[incorrect])))

