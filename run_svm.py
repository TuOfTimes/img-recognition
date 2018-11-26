from dataProcessor import ImageFileHandler
from Classifiers.SVM.lsvc import LinearSupportVectorClassifier
import logging
logging.basicConfig(filename="svm.log",level=logging.INFO)

def logging_wrapper(func):
    def inner(*args, **kwargs):
        try:
            func(*args, **kwargs)
        except Exception as e:
            logging.exception("There was an exception {} in function {}".format(str(e),str(func)))
    return inner



@logging_wrapper
def main():
    f = open("LinearSVCDetails.txt","w")

    DataPath = "Data/Processed/"

    imf = ImageFileHandler(DataPath + "train_m50_p5_a0.npy", y_index=0)
    lsvc = LinearSupportVectorClassifier(imf.xMatrix, imf.yVector)
    f.write("Data Loaded and Classifier initialized\n")
    f.write("Starting hyper-parameter tuning\n")

    best_params, best_score, results = lsvc.find_best_params()

    f.write("The best hyper-parameters are as follows: \n")
    f.write("C: {}\t| tol: {} with an F1-Measure of {}\n\n".format(
        best_params['C'], best_params['tol'], best_score
    ))

    f.write("\nPerformance metrics for the first 100 hyper-parameters_tested:\n\n")
    index = 0
    while (index < 100 and index < len(results['params'])):
        f.write("C: {}\t| tol: {} --> {}\n".format(
            results['params'][index]['C'],
            results['params'][index]['tol'],
            results['mean_test_score'][index]
        ))
        index += 1

    f.write("\n\nInitializing and training a Linear Support Vector Classifier with C={} and tol={} \n".format(
        best_params['C'], best_params['tol']))
    best_C = float(best_params['C'])
    best_tol = float(best_params['tol'])
    lsvc = LinearSupportVectorClassifier(imf.xMatrix, imf.yVector)
    lsvc.initialize_classifier(tol=best_tol, C=best_C)
    lsvc.train()


    imf_test = ImageFileHandler(DataPath + "test_m50_p5_a0.npy")
    predictions = lsvc(imf_test.XMatrix)
    f.write("Creating Submissions file\n")

    import csv
    with open("Data/Raw/categories.csv", mode='r') as infile:
        reader = csv.reader(infile)
        categories = {i: row[0] for i, row in enumerate(reader)}


    with open("submissions.txt", 'w') as file:
        file.write('Id,Category\n')
        for i, prediction in enumerate(predictions):
            file.write(str(i) + ',' + categories[prediction])
            file.write('\n')

    f.write("Done!\n")
    f.close()

if __name__ == "__main__":
    main()