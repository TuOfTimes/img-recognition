from dataProcessor import ImageFileHandler
import logging
import math
import csv
import random
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy
logging.basicConfig(filename="neuralnet.log",level=logging.INFO)


def logging_wrapper(func):
    def inner(*args, **kwargs):
        try:
            func(*args, **kwargs)
        except Exception as e:
            logging.exception("There was an exception {} in function {}".format(str(e),str(func)))
    return inner


def initialize_network(nb_inputs, nb_hidden_layer, nb_hidden_node, nb_outputs):
    network = list()
    hidden_layer = [{'weights': [random.uniform(0, 0.01) for i in range(nb_inputs + 1)]} for i in range(nb_hidden_node)]
    network.append(hidden_layer)
    for i in range(nb_hidden_layer - 1):
        hidden_layer = [{'weights':[random.uniform(0,0.1) for i in range(nb_hidden_node + 1)]} for i in range(nb_hidden_node)]
        network.append(hidden_layer)
    output_layer = [{'weights':[random.uniform(0,0.1) for i in range(nb_hidden_node + 1)]} for i in range(nb_outputs)]
    network.append(output_layer)
    return network


def split_data(X, Y):
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.4, random_state=1)
    X_train,X_valid, Y_train, Y_valid = train_test_split(X_train, Y_train, test_size=0.2, random_state=1)
    return X_train, X_valid, X_test, Y_train, Y_valid,Y_test


# neuron activation for one input
def activate_neuron(weights, inputs):
    activation = weights[-1]  # bias term
    for i in range(len(weights)-2):
        activation += weights[i]*inputs[i]
    return activation


def sigmoid_derivate(x):
    # derivative of sigmoid function if f(x)*(1-f(x))
    return x * (1.0-x)


def sigmoid(activation):
    # sigmoid activation function
    return 1.0/(1.0 + math.exp(-activation))


def feed_forward(network, row):
    inputs = row
    for layer in network:
        new_inputs = []
        for neuron in layer:
            activation = activate_neuron(neuron['weights'], inputs)
            neuron['output'] = sigmoid(activation)
            new_inputs.append(neuron['output'])
        inputs = new_inputs
    return inputs


def backward_propagate_error(network, Y_train):

    for i in reversed(range(len(network))):
        layer = network[i]
        errors = list()
        if(i!=len(network)-1):
            # ie this is a hidden layer
            for j in range(len(layer)):
                error = 0.0
                for neuron in network[i+1]:
                    error += (neuron['weights'][j]*neuron['delta'])
                errors.append(error)

        else:
            # ie this is the output layer
            for j in range(len(layer)):
                neuron = layer[j]
                errors.append(Y_train[j] - neuron['output'])

        for j in range(len(layer)):
            neuron = layer[j]
            neuron['delta'] = errors[j] * sigmoid_derivate(neuron['output'])


def update_weights(network, row, l_rate):
    for i in range(len(network)):
        inputs = row[:-1]
        if i != 0:
            inputs = [neuron['output'] for neuron in network[i - 1]]
        for neuron in network[i]:
            for j in range(len(inputs)):
                neuron['weights'][j] += l_rate * neuron['delta'] * inputs[j]
            neuron['weights'][-1] += l_rate * neuron['delta']


def train_network(network, X_train, Y_train,l_rate, nb_epoch, nb_outputs):
    for epoch in range(nb_epoch):
        sum_error = 0
        random.shuffle(X_train)
        for i in range(0, len(X_train)): # we go through every minibatch
            row = X_train[i]
            outputs = feed_forward(network, row[:len(row)-1])

            expected = [0 for i in range(nb_outputs)]
            expected[int(Y_train[i])] = 1
            #print(expected)
            #print(outputs)
            sum_error += sum([math.pow((expected[i]-outputs[i]) , 2) for i in range(len(expected))]) # loss function
            backward_propagate_error(network, Y_train)
            update_weights(network, row, l_rate)
        print('Current Epoch: %d, Learning Rate: %.3f, Error: %.3f' % (epoch, l_rate, sum_error))




def run_network(X,Y):

    X_train, X_valid, X_test, Y_train, Y_valid, Y_test = split_data(X, Y)
    file = open("nn_results.csv", 'w')
    writer = csv.writer(file)
    writer.writerow(["alpha_value", "nb_hidden_layers", "nb_hiddenNodes", "validation_accuracy"])
    for i in range(len(X_train)):
        # X_train[i].append(1) # for the bias term
        numpy.append(X_train[i],(Y_train[i]))
    # for i in range(len(X_valid)):
    #     X_valid[i].append(1)
    # for i in range(len(X_test)):
    #     X_test[i].append(1)

    max_accuracy = 0
    optimal_params = []

    for alpha in [0.01, 0.05, 0.1, 0.5, 0.7, 0.9]:
        for nb_hidden_layer in [1, 2]:
            for nb_hidden_node in [50, 100]:
                parameters = [alpha, nb_hidden_layer, nb_hidden_node]
                predictions = []
                print("alpha : " + str(alpha) + " hidden layers : " + str(nb_hidden_layer) + " nodes : " + str(nb_hidden_node))
                logging.info("alpha : " + str(alpha) + " hidden layers : " + str(nb_hidden_layer) + " nodes : " + str(nb_hidden_node))

                network = initialize_network(len(X_train[0]) - 1, nb_hidden_layer, nb_hidden_node, 31)
                train_network(network, X_train,Y_train, alpha, 4, 31)
                for i in range(len(X_valid)):
                    output = feed_forward(network, X_valid[i])
                    predictions.append(output.index(max(output)))
                accuracy = accuracy_score(Y_valid, predictions)
                print("accuracy: " +accuracy+'params:' + str(parameters[0]) +','+ str(parameters[1])+',' + str(parameters[2]))
                logging.info("accuracy: "+accuracy+'params:' + str(parameters[0]) +','+ str(parameters[1])+',' + str(parameters[2]))
                if accuracy > max_accuracy:
                    max_accuracy = accuracy
                    optimal_params = parameters
                parameters.append(accuracy)
                writer.writerow(parameters)

    network = initialize_network(len(X_train[0])-1, optimal_params[1], optimal_params[2], 31)
    train_network(network, X_train,Y_train, optimal_params[0], 4, 31)
    predictions = []
    output = []
    for i in range(len(X_test)):
        output = feed_forward(network, X_test[i])
        predictions.append(output.index(max(output)))

    accuracy = accuracy_score(Y_test, predictions)
    optimal_params.append(accuracy)
    writer.writerow(optimal_params)

    file.close()




@logging_wrapper
def main3():
    f = open("NeuralNetDetails.txt","w")
    DataPath = "Data/Processed/"

    imf = ImageFileHandler(DataPath + "train_m50_p5_a0.npy", y_index=0)
    X=imf.xMatrix
    print(X)
    Y= imf.yVector
    run_network(X,Y)

    #network= initialize_network(nb_inputs, nb_hidden_layer, nb_hidden_node, nb_outputs)
    #lsvc = LinearSupportVectorClassifier(imf.xMatrix, imf.yVector)
    f.write("Data Loaded and Classifier initialized\n")
    f.write("Starting hyper-parameter tuning\n")

    #train_network(network, X_train, Y_train, l_rate, nb_epoch, nb_outputs)
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


f = open("NeuralNetDetails.txt","w")
DataPath = "Data/Processed/"

imf = ImageFileHandler(DataPath + "train_m50_p5_a0.npy", y_index=0)
X=imf.xMatrix
print(X[1])
Y= imf.yVector
print(Y)

run_network(X,Y)
