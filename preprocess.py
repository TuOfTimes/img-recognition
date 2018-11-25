#!/usr/bin/env python
# coding: utf-8

# In[7]:


# import labels and images from file

import numpy as np
import csv


with open('Data/categories.csv', mode='r') as infile:
    reader = csv.reader(infile)
    categories = {row[0]:i for i, row in enumerate(reader)}
    
with open('Data/train_labels.csv', mode='r') as infile:
    next(infile)
    reader = csv.reader(infile)
    labels = {int(row[0]):categories[row[1]] for row in reader}   
# separate images by labels
images = np.load('Data/train_images.npy', encoding='bytes')
for i, image in enumerate(images):
    images[i][1] = image[1].reshape(100, 100)
    label = labels[images[i][0]]
    images[i][0] = label


# In[8]:


X = [item[1] for item in images]
y = [item[0] for item in images]


# In[9]:


# preprocess image data
from skimage.measure import label,regionprops
from skimage.transform import resize

for i, img in enumerate(X):
    
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
        X[i] = np.zeros((50,50))
    else:
        
        selected_img = img[min(coords[0],coords[2]):max(coords[0],coords[2]),min(coords[1],coords[3]):max(coords[1],coords[3])]

        X[i] = resize(selected_img,(50,50))


# In[10]:


output = []
for label, image in zip(y, X):
    output.append(np.array([label, image.astype('float64')], dtype='object'))


# In[11]:


output = np.array(output)


# In[12]:


output[0]


# In[13]:


np.save('Data/processed_train_images.npy', output)


# In[ ]:




