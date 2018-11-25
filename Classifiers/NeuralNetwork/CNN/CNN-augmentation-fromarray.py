#!/usr/bin/env python
# coding: utf-8

# In[1]:

import numpy as np
import sys
sys.path.append("../../../")
from dataProcessor import ImageFileHandler
from keras.utils import np_utils
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K

# CHANGE THESE IF YOU WANT
image_src = '../../../Data/Processed/train_data_non_bin.npy'

epochs = 3
img_width, img_height = 50,50
batch_size = 64

save_weights = 'save.h5'
output_file = '5050_padding0_epochs3.txt'

# 
f = open(output_file, 'w')

# import labels and images from file
images = ImageFileHandler(image_src)


# In[2]:
X = images.xMatrix
y = images.yVector

X_array = np.array(X)
f.write("%s: %s" % ("input shape", X_array.shape))
X_array = X_array.reshape(X_array.shape[0], img_width, img_height, 1)
X_array = X_array.astype('float64')
X_array /= 255

y_array = np.array(y)
y_array = np_utils.to_categorical(y_array, 31)


# In[3]:
y_array[0]


# In[4]:
# train-validation split
X_other, X_val, y_other, y_val = train_test_split(X_array, y_array, test_size=0.20, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X_other, y_other, test_size=0.25, random_state=24)


# In[5]:
f.write("%s: %s\n" % ("training points: ", len(X_train)))
f.write("%s: %s\n" % ("validation points: ", len(X_val)))
f.write("%s: %s\n" % ("test points: ", len(X_test)))


# In[8]:
# display first 9 images and their classes
# for i in range(9):
#     plt.subplot(3,3,i+1)
#     plt.imshow(X_train[i].reshape(50,50), cmap='gray', interpolation='none')
#     plt.title("Class {}".format(np.argmax(y_train[i])))
# plt.show()


# In[9]:

nb_train_samples = 6000
nb_validation_samples = 2000
input_shape = (img_width, img_height, 1)


# In[10]:


model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=input_shape))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3)))
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

model.save_weights(save_weights)


# In[57]:


score = model.evaluate(X_test, y_test, verbose=0)
f.write("%s: %s\n" % ("Test score", score[0]))
f.write("%s: %s\n" % ("Test accuracy", score[1]))


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

f.close()