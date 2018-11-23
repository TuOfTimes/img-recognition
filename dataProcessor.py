import numpy as np
import matplotlib.pyplot as plt
from skimage.measure import label,regionprops
from skimage.transform import resize

class ImageProcessor():

    def __init__(self,filePath,labelPath=None,categoryPath=None):
        self.images = np.load(filePath,encoding='bytes')
        self.labels = []
        self.categories = {}
        if labelPath and categoryPath:
            with open(labelPath,"r") as f:      #store in list self.labels all the labels with index of list corresponding to index of image
                data = f.readlines()
            for i in range(1,len(data)):
                self.labels.append(data[i].split(',')[1].strip("\n"))
            self.categories = {}

            with open(categoryPath,"r") as f:
                data = f.readlines()
            for i in range(0,len(data)):
                self.categories[data[i].strip('\n')] = i
            print(self.categories)

    def getImage(self,imageIndex,reshape=False):
        if reshape:
            return self.images[imageIndex][1].reshape(100,100)
        return self.images[imageIndex][1]


    def getID(self,imageIndex):
        return self.images[imageIndex][0]

    def showImageAtIndex(self,imageIndex):
        img = self.getImage(imageIndex,True)
        plt.imshow(img)
        plt.show()

    def getProcessedImage(self,imageIndex,mask_val=100,dims=(50,50),binaryRepresentation=False,min_area=100, padding_to_add=0):
        '''
        Sets all pixels above mask value to 255 and those below to 0 then
        Creates bounding boxes around each connected set of pixels
        Returns the biggest box
        '''
        img = self.getImage(imageIndex,reshape=True)
        masked_img = self.mask_image(img,mask_val)

        coords = self.getBiggestImageBoundingBox(masked_img)

        height = abs(coords[3] - coords[1])
        width = abs(coords[2] - coords[0])
        area = height * width
        if area < min_area:     #if bounding box area is less than the minimum area specified assume empty image
            return np.zeros(dims)


        if binaryRepresentation:
            selected_img = masked_img[min(coords[0],coords[2]):max(coords[0],coords[2]),min(coords[1],coords[3]):max(coords[1],coords[3])]
            for i in range(0, len(selected_img)):
                for j in range(0,len(selected_img[i])):
                    if selected_img[i][j] != 0:
                        selected_img[i][j] = 1
        else:
            selected_img = img[min(coords[0],coords[2]):max(coords[0],coords[2]),min(coords[1],coords[3]):max(coords[1],coords[3])]

        return resize(selected_img,dims)


    def mask_image(self,img,mask_val):
        img_copy = img.copy()
        for i in range(0, len(img_copy)):
            for j in range(0, len(img_copy[i])):
                if img_copy[i][j] < mask_val:
                    img_copy[i][j] = 0
                else:
                    img_copy[i][j] = 255
        return img_copy

    def getBiggestImageBoundingBox(self,img):
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
        return coords

    def removeNoise(self,img):
        bbox = self.getBiggestImageBoundingBox(img)
        for i in range(0,len(img)):
            for j in range(0,len(img[i])):
                if i<bbox[0] or i>bbox[2] or j<bbox[1] or j>bbox[3]:
                    img[i][j] = 0

    def getProcessedImages(self,mask_val=50,dims=(50,50),binaryRepresentation=False,min_area=100):
        processed_images = []
        for i in range(0,len(self.images)):
            img = self.getProcessedImage(i,mask_val=mask_val,dims=dims,binaryRepresentation=binaryRepresentation,min_area=min_area)
            img = img.reshape(dims[0]*dims[1])
            processed_images.append(img)
        return processed_images

    def labelImages(self,images):
        if not self.categories or not self.labels:
            return None

        output = []
        if len(images) != len(self.labels):
            return output

        for i in range(0,len(self.labels)):
            temp = np.ndarray.tolist((images[i]))
            temp = [self.categories[self.labels[i]]] + temp
            output.append(np.asarray(temp))
            # output.append([self.categories[self.labels[i]]])
            # output[i].append(np.ndarray.tolist(images[i]))
        return output

    @staticmethod
    def showImage(img,toReshape=False,reshape_dims =(100,100)):
        if toReshape:
            img = img.reshape(reshape_dims[0],reshape_dims[1])
        plt.imshow(img)
        plt.show()

class ImageFileHandler():
    def __init__(self,imagePath,y_index=-1):
        self.images = np.load(imagePath,encoding="bytes")
        self.yVector = []
        self.xMatrix = self.images
        if y_index == 0:
            self.yVector = self._getYVector_(y_index)
            self.xMatrix = self._getXVector_(y_index)

    def _getYVector_(self,y_index):
        y_vec = []
        for i in range(0,len(self.images)):
            y_vec.append(self.images[i][y_index])
        return y_vec

    def _getXVector_(self,y_index):
        x_matrix = np.delete(self.images,y_index,1)
        return x_matrix

    @staticmethod
    def saveNPYFile(img,img_path):
        if img_path[-4:] != ".npy":
            print("Invalid File Path")
            return
        np.save(img_path,img)


def processTrainingData():
    DATA_PATH = "Data/Raw/"
    DATA_PROCESSED_PATH = "Data/Processed/"

    img_processor = ImageProcessor(filePath=DATA_PATH + "train_images.npy",
                                   labelPath=DATA_PATH + "train_labels.csv",
                                   categoryPath=DATA_PATH + "categories.csv")

    # Create Non-Binary representation
    processed_imgs_nb = img_processor.getProcessedImages(mask_val=50,
                                                         dims=(50, 50),
                                                         binaryRepresentation=False,
                                                         min_area=100)

    labelled_imgs_nb = img_processor.labelImages(processed_imgs_nb)
    ifm = ImageFileHandler.saveNPYFile(labelled_imgs_nb, "Data/Processed/train_data_non_bin.npy")

    # Create Binary Representation
    processed_imgs_b = img_processor.getProcessedImages(mask_val=50,
                                                        dims=(50, 50),
                                                        binaryRepresentation=True,
                                                        min_area=100)

    labelled_imgs_b = img_processor.labelImages(processed_imgs_b)
    ImageFileHandler.saveNPYFile(labelled_imgs_b, "Data/Processed/train_data_bin.npy")

def processTestData():
    DATA_PATH = "Data/Raw/"

    img_processor = ImageProcessor(filePath=DATA_PATH + "test_images.npy")

    # Create Non-Binary representation
    processed_imgs_nb = img_processor.getProcessedImages(mask_val=50,
                                                         dims=(50, 50),
                                                         binaryRepresentation=False,
                                                         min_area=100)

    ifm = ImageFileHandler.saveNPYFile(processed_imgs_nb, "Data/Processed/test_data_non_bin.npy")

    # Create Binary Representation
    processed_imgs_b = img_processor.getProcessedImages(mask_val=50,
                                                        dims=(50, 50),
                                                        binaryRepresentation=True,
                                                        min_area=100)

    ImageFileHandler.saveNPYFile(processed_imgs_b, "Data/Processed/test_data_bin.npy")


if __name__ == "__main__":
    # DATA_PROCESSED_PATH = "Data/Processed/"
    # #processTrainingData()
    #
    #
    # imf_nb = ImageFileHandler(imagePath=DATA_PROCESSED_PATH+"train_data_bin.npy",
    #                           y_index=0)
    # img = imf_nb.xMatrix[31]
    # ImageProcessor.showImage(img,True,(50,50))
    # print(imf_nb.yVector[31])
    # # img_processor = ImageProcessor("Data/Raw/train_images.npy")
    # img = img_processor.getProcessedImage(imageIndex=31,mask_val=50,min_area=125)
    # img_processor.showImage(img)

    #processTestData()
    imf = ImageFileHandler(imagePath="Data/Processed/test_data_non_bin.npy")
    ImageProcessor.showImage(imf.xMatrix[5],True,(50,50))

    imp = ImageProcessor("Data/Raw/test_images.npy")
    imp.showImageAtIndex(5)