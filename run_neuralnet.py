from dataProcessor import ImageFileHandler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import math
import logging
import random
import os

logging.basicConfig(filename="neuralnet.log",level=logging.INFO)


def logging_wrapper(func):
    def inner(*args, **kwargs):
        try:
            func(*args, **kwargs)
        except Exception as e:
            logging.exception("There was an exception {} in function {}".format(str(e),str(func)))
    return inner


def split_data(X, Y):
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.4, random_state=1)
    X_train,X_valid, Y_train, Y_valid = train_test_split(X_train, Y_train, test_size=0.2, random_state=1)
    return X_train, X_valid, X_test, Y_train, Y_valid,Y_test


#Creates network for a given architecture
def initialize_network(n_inputs, nb_idden_layer, nb_idden_node, nb_outputs):
    network = list()
    hidden_layer = [{'weights': [random.uniform(0, 0.01) for i in range(n_inputs + 1)]} for i in range(nb_idden_node)]
    network.append(hidden_layer)
    for i in range(nb_idden_layer - 1):
        hidden_layer = [{'weights':[random.uniform(0,0.1) for i in range(nb_idden_node + 1)]} for i in range(nb_idden_node)]
        network.append(hidden_layer)
    output_layer = [{'weights':[random.uniform(0,0.1) for i in range(nb_idden_node + 1)]} for i in range(nb_outputs)]
    network.append(output_layer)
    return network


#Neuron activation function
def activation_func(weights, inputs):
    activation = weights[-1]
    for i in range(len(weights)-1):
        activation += weights[i]*inputs[i]
    return activation


def sigmoid(x):
    return 1.0/(1.0 + math.exp(-x))


def sigmoid_derivative(x):
    return x * (1.0-x)


def feed_forward(network, row):
    inputs = row
    for layer in network:
        temp = []
        for n in layer:
            activate = activation_func(n['weights'], inputs)
            n['output'] = sigmoid(activate)
            temp.append(n['output'])
        inputs = temp
    return inputs


def backpropagation(network, true_output):

    for i in reversed(range(len(network))):
        layer = network[i]
        list_errors = list()
        #if output layer
        if i == len(network)-1:
            for j in range(len(layer)):
                n = layer[j]
            list_errors.append(true_output[j] - n['output'])

        #not output layer = hidden layer
        else:
            for j in range(len(layer)):
                error = 0.0
            for n in network[i+1]:
                error += (n['weights'][j]*n['delta'])
            list_errors.append(error)
        
        for j in range(len(layer)):
            n = layer[j]
            n['delta'] = list_errors[j] * sigmoid_derivative(n['output'])


def update_weights(network, row, learning_rate):
    for i in range(len(network)):
        inputs = row[:-1]
        if i != 0:
            inputs = [n['output'] for n in network[i - 1]]
        for n in network[i]:
            for j in range(len(inputs)):
                n['weights'][j] += learning_rate * n['delta'] * inputs[j]
            n['weights'][-1] += learning_rate * n['delta']


def train_network(network, X_train,Y_train,learning_rate, nb_epoch, nb_outputs):
    for epoch in range(nb_epoch):
        total_error = 0
        random.shuffle(X_train)
        for i in range(0, len(X_train)):
            row = X_train[i]
            outputs = feed_forward(network, row[:len(row)-1])
            true_output = [0 for i in range(nb_outputs)]
            true_output[int(Y_train[i])] = 1
            total_error += sum([math.pow((true_output[i]-outputs[i]) , 2) for i in range(len(true_output))])
            backpropagation(network, true_output)
            update_weights(network, row, learning_rate)
            
        print('Cur epoch:' + str(epoch)+' Learning rate: '+ str(learning_rate)+'Error function'+ str(total_error))
        logging.info('Cur epoch:' + str(epoch)+' Learning rate: '+ str(learning_rate)+'Error function'+ str(total_error))


def crossvalidation(X, Y):
    X_train, X_valid, X_test, Y_train, Y_valid, Y_test = split_data(X, Y)
    max_accuracy = 0
    optimal_architecture = []
    for learning_rate in [0.05, 0.1, 0.3, 0.5, 0.7, 0.9]:
        for nb_idden_layer in [1, 2]:
            for nb_idden_node in [50, 100]:
                architecture = [learning_rate, nb_idden_layer, nb_idden_node]
                predictions = []
                print("learning rate : " + str(learning_rate) + " number of layers: " + str(nb_idden_layer) + " neurons per layer : " + str(nb_idden_node))
                logging.info(("learning_rate : " + str(learning_rate) + " number of layers : " + str(nb_idden_layer) + " neurons per layer : " + str(nb_idden_node)))
                network = initialize_network(len(X_train[0]) - 1, nb_idden_layer, nb_idden_node, 31)
                train_network(network, X_train,Y_train, learning_rate, 3, 31)
                for i in range(len(X_valid)):
                    output = feed_forward(network, X_valid[i])
                    predictions.append(output.index(max(output)))
                accuracy = accuracy_score(Y_valid, predictions)
                print("accuracy for architecture " + str(architecture[0]) + str(architecture[1]) + str(architecture[2]))
                print(accuracy)
                logging.info("accuracy for architecture " + str(architecture[0]) + str(architecture[1]) + str(architecture[2]))
                logging.info(accuracy)
                if accuracy > max_accuracy:
                    max_accuracy = accuracy
                    optimal_architecture = architecture

    network = initialize_network(len(X_train[0])-1, optimal_architecture[1], optimal_architecture[2], 31)
    train_network(network, X_train,Y_train, optimal_architecture[0], 5, 31)
    predictions = []

    for i in range(len(X_test)):
        output = feed_forward(network, X_test[i])
        predictions.append(output.index(max(output)))

    accuracy = accuracy_score(Y_test, predictions)
    optimal_architecture.append(accuracy)
    logging.info("optimal architecture : ")
    logging.info(optimal_architecture)


#loop over all the different input files to train neural network on differently preprocessed datasets

DataPath = "Data/Processed/"
for filename in os.listdir(DataPath):
    logging.info(filename)
    imf = ImageFileHandler(DataPath + filename, y_index=0)
    X=imf.xMatrix
    Y= imf.yVector
    crossvalidation(X,Y)
