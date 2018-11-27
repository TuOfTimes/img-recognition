# Image Recognition

## Introduction
In this project, we use a training set of 10,000 images to train multiple classifying models. Depending on the performance with the test data (also 10,000 images),
we will pick the best model to compete in a Kaggle Competition.


## Parameters We Are Testing

### FFNN
* Learning Rate : [0.01, 0.05, 0.1, 0.5, 0.7, 0.9]
* Number of layers :[1, 2]
* Neurons per layer [50, 100]:
* Padding Around Images - [0,5,10]
* Mask Values - [1,50,100]


### CNN
* Epochs - [100, 200]
* Batch Size - [32,64]
* Padding Around Images - [0,5,10]
* Mask Values - [1,50,100]

### SVM
* C
* tol


## Submissions:
* e200_b32_m_1_p5_a0
* e200_b64_m1_p5_a0
* e100_b64_m_50_p0_a0 -> Highest Score
* e200_b32_m_1_p5_a0
* alexnet_e50_b32_m_50_p0_a0
