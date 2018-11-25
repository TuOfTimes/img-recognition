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
import logging

'''
    Change model by editing the number of layers in the create_model method
'''

logging.basicConfig(filename="q2.log",level=logging.INFO)

def logging_wrapper(func):
    def inner(*args, **kwargs):
        try:
            func(*args, **kwargs)
        except Exception as e:
            logging.exception("There was an exception {} in function {}".format(str(e),str(func)))
    return inner

@logging_wrapper
def create_submission(test_src,categories_csv,weight_file,img_width,img_height,output_csv):
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


    model = create_model()


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

@logging_wrapper
def create_model():
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
    
    return model

@logging_wrapper
def run_CNN(image_src,epochs,batch_size,img_width,img_height,save_weights,output_file):
    f = open(output_file, 'w')

    # import labels and images from file
    images = ImageFileHandler(image_src,y_index=0)

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


    model = create_model()



    # this is the augmentation configuration we will use for training
    train_datagen = ImageDataGenerator(
        rotation_range=45,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)


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

    score = model.evaluate(X_test, y_test, verbose=0)
    f.write("%s: %s\n" % ("Test score", score[0]))
    f.write("%s: %s\n" % ("Test accuracy", score[1]))

    f.close()


if __name__ == "__main__":
    epochs_list = (100,200)
    batch_size_list = (16,32,64)
    mask_val_list = (1,50,100)
    padding_val_list = (0,5,10)
    min_area_list = (0,100)

    for epochs in epochs_list:
        for batch_size in batch_size_list:
            for mask_val in mask_val_list:
                for padding_val in padding_val_list:
                    for min_area in min_area_list:
                        logging.info("E-{}, B-{}, M-{}, P-{}, A-{}".format(epochs,batch_size,mask_val,padding_val,min_area))

                        image_src = '../../../Data/Processed/train_m{}_p{}_a{}.npy'.format(mask_val,padding_val,min_area)

                        img_width, img_height = 50, 50

                        save_weights = 'Outputs/Weights/e{}_b{}_m_{}_p{}_a{}.h5'.format(epochs,batch_size,mask_val,padding_val,min_area)
                        output_file = 'Outputs/TestScores/e{}_b{}_m_{}_p{}_a{}.txt'.format(epochs,batch_size,mask_val,padding_val,min_area)

                        logging.info("Running CNN")
                        run_CNN(image_src=image_src,
                                epochs=epochs,
                                img_width=img_width,
                                img_height=img_height,
                                batch_size = batch_size,
                                save_weights=save_weights,
                                output_file=output_file)


                        test_src = '../../../Data/Processed/test_m{}_p{}_a{}.npy'.format(mask_val,padding_val,min_area)
                        categories_csv = '../../../Data/Raw/categories.csv'
                        weight_file = save_weights
                        output_csv = 'Outputs/Submissions/e{}_b{}_m_{}_p{}_a{}.csv'.format(epochs,batch_size,mask_val,padding_val,min_area)


                        logging.info("Creating Submissions")
                        create_submission(test_src=test_src,
                                          categories_csv=categories_csv,
                                          img_height=img_height,
                                          img_width=img_width,
                                          weight_file=weight_file,
                                          output_csv=output_csv
                                          )