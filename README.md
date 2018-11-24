# Image Recognition

## Introduction
In this project, we use a training set of 10,000 images to train multiple classifying models. Depending on the performance with the test data (also 10,000 images),
we will pick the best model to compete in a Kaggle Competition.


## Parameters We Are Testing

### CNN
* Epochs - [100, 200]
* Batch Size - [32,64]
* Padding Around Images - [0,5,10]
* Mask Values - [1,50,100]

### SVM
* C
* tol