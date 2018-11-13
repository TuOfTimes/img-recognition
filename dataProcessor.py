import numpy as np
import matplotlib.pyplot as plt


class Data():
    def __init__(self,filePath,labelPath,categoryPath):
        self.images = np.load(filePath,encoding='bytes')
        self.labels = []

        with open(labelPath,"r") as f:      #store in list self.labels all the labels with index of list corresponding to index of image
            data = f.readlines()
        for i in range(1,len(data)):
            self.labels.append(data[i].split(',')[1].strip("\n"))

        self.categories = {}

        with open(categoryPath,"r") as f:
            data = f.readlines()
        for i in range(0,len(data)):
            self.categories[data[i]] = i


    def getImage(self,imageIndex,reshape=False):
        if reshape:
            return self.images[imageIndex][1].reshape(100,100)
        return self.images[imageIndex][1]

    def getID(self,imageIndex):
        return self.images[imageIndex][0]

    def showImage(self,imageIndex):
        img = self.getImage(imageIndex,True)
        plt.imshow(img)
        plt.show()

    def createLabelledTraining(self,savingPath):
        '''
        Saves a new .npy file at the saving path containing a 2D array:
        - 1st index -> class ID (0 - 31)
        - 2nd index ->vector containing image data (10000 points)
        '''
        pass

if __name__ == "__main__":
    data = Data(filePath="Data/train_images.npy",labelPath="Data/train_labels.csv",categoryPath="Data/categories.csv")


