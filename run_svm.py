from dataProcessor import ImageFileHandler
from SVM.lsvc import LinearSupportVectorClassifier

if __name__ == "__main__":
    DataPath = "Data/Processed/"
    imf_bin = ImageFileHandler(DataPath+"train_data_bin.npy")
    imf_non_bin = ImageFileHandler(DataPath+"train_data_non_bin.npy")

    ls_bin = LinearSupportVectorClassifier(imf_bin.xMatrix,imf_bin.yVector)
    ls_non_bin = LinearSupportVectorClassifier(imf_non_bin.xMatrix,imf_non_bin.yVector)

