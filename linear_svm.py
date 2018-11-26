from dataProcessor import ImageFileHandler
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import logging
import csv

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

    train_results = []
    valid_results = []

    imf = ImageFileHandler(DataPath + "train_m50_p5_a0.npy", y_index=0)
    X_other, X_val, y_other, y_val = train_test_split(imf.xMatrix, imf.yVector, test_size=0.20, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X_other, y_other, test_size=0.25, random_state=24)

    c_vals = [0.001, 0.01, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0]
    for c in c_vals:
        logging.info("Testing C value: {}".format(c))
        lsvc = LinearSVC(C=c, random_state=0, loss='hinge', multi_class='ovr', max_iter=1000)
        lsvc.fit(X_train, y_train)

        y_pred_train = lsvc.predict(X_train)
        y_pred_valid = lsvc.predict(X_val)

        training_acc = accuracy_score(y_train,y_pred_train)
        val_acc = accuracy_score(y_val,y_pred_valid)

        train_results.append(training_acc)
        valid_results.append(val_acc)

        logging.info("Training score is {}".format(training_acc))
        logging.info("Validation score is {}".format(val_acc))

    f.write("\nPerformance metrics for the hyper-parameters tested:\n\n")
    index = 0
    while (index < 100 and index < len(training_acc)):
        f.write("C: {}\t| training_accuracy: | validation_accuracy: \n".format(
            c_vals[index],
            train_results[index],
            valid_results[index]))
        index += 1

if __name__ == "__main__":
    main()