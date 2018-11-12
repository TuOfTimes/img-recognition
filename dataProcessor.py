import numpy as np
import matplotlib.pyplot as plt


class Data():
    def __init__(self,filePath):
        self.filePath = filePath
        self.images = np.load(filePath,encoding='bytes')

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

if __name__ == "__main__":
    data = Data("Data/train_images.npy")
    print(data.getImage(1,True))
    print(data.ID(1))
    data.showImage(1)


