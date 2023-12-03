# -*- coding: utf-8 -*-
"""
Created on Tue Mar  9 23:57:11 2021
Machine Learning
exercise NN 2023
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
import pandas as pd
import time


def sigmoid(Z):
    """
    Compute the sigmoid of Z
    Input Arguments:
    Z - A scalar or numpy array of any size.
    Return: A - sigmoid(Z)
    """
    A = 1/(1+np.exp(-Z))
    
    return A

def ReLU(Z):
    """
    Compute ReLU(Z)
    Input Arguments:
    Z - A scalar or numpy array of any size.
    Return:
    A - ReLU(Z)
    """
    A = np.maximum(0, Z)
    
    return A

def dReLU(Z):
    """
    Compute dReLU(Z)
    Input Arguments:
    Z - A scalar or numpy array of any size.
    Return:
    A - dReLU(Z)
    """
    A = (Z > 0) * 1
    
    return A
      

def init_parameters(Lin, Lout):
    """
    Init_parameters randomly initialize the parameters of a layer with Lin
    incoming inputs and Lout outputs 
    Input arguments: 
    Lin - the number of incoming inputs to the layer (not including the bias)
    Lout - the number of output connections 
    Output arguments:
    Theta - the initial weight matrix, whose size is Lout x Lin+1 (the +1 is for the bias).    
    Usage: Theta = init_parameters(Lin, Lout)
    
    """
    
    factor = np.sqrt(6/(Lin+Lout))
    Theta = np.zeros((Lout, Lin + 1))
    Theta = 2 * factor * (np.random.rand(Lout, Lin + 1) - 0.5)
    return Theta


def ff_predict(weights, X, y):
    """
    ff_predict employs forward propagation on a multi-layer network and
    determines the labels of the inputs
    Input arguments
    weights - list of matrices of parameters (weights) between layers
    X - input matrix
    y - input labels
    Output arguments:
    p - the predicted labels of the inputs
    detectp - detection percentage
    Usage: p, detectp = ff_predict(weights, X, y)
    """
    m = X.shape[0]
    num_layers = len(weights)
    num_outputs = weights[-1].shape[0]
    p = np.zeros((m, 1))
    a = [X]

    for i in range(num_layers):
        a[i] = np.concatenate((np.ones((a[i].shape[0], 1)), a[i]), axis=1)
        z = np.dot(a[i], weights[i].T)
        if i == num_layers - 1:
            a.append(sigmoid(z))
        else:
            a.append(ReLU(z))

    p = np.argmax(a[-1].T, axis=0)
    p = p.reshape(p.shape[0], 1)
    detectp = np.sum(p == y) / m * 100

    return p, detectp

    


def backprop(weights, numOfLayers, X, y, max_iter = 1000, learningRate = 0.9, Lambda = 0):
    """
    backprop - BackPropagation for training a neural network
    Input arguments
    weights - the array of all the weights
    numOfHiddenLayers- the number of hidden layers ie the lenght of weights
    X - input matrix
    y - labels of the input examples
    max_iter - maximum number of iterations (epochs).
    alpha - learning coefficient.
    Lambda - regularization coefficient.
    
    Output arguments
    cost - the cost function
    Theta1 - updated weight matrix between the input and the first 
        hidden layer
    Theta2 - updated weight matrix between the hidden layer and the output 
        layer (or a second hidden layer)
    
    Usage:
    [cost,Theta1,Theta2] = backprop(Theta1, Theta2, X,y,max_iter, alpha,Lambda)
    """
    
    num_of_training_examples = X.shape[0]
    deltas = [np.zeros_like(w) for w in weights]
    cost = 0
    weightsGrad = []
    weightsDer = []
    for i in range(numOfLayers):
        weightsGrad.append(np.zeros(weights[i].shape))
        weightsDer.append(np.zeros(weights[i].shape))
    for iter in range(max_iter):
        cost = 0
        for i in range(numOfLayers):

            weightsGrad[i] = np.zeros(weights[i].shape)
            weightsDer[i] = np.zeros(weights[i].shape)
        r = np.random.permutation(num_of_training_examples)
        
        for k in range(num_of_training_examples):
            trainingExample = X[r[k],:]
            trainingExample = trainingExample.reshape(1, trainingExample.shape[0])
            ### Forward propagation
            bias = np.ones((trainingExample.shape[0], 1))
            trainingExample = np.concatenate((bias, trainingExample), axis=1)
            trainingExample = trainingExample.T
            a = [trainingExample]
            for i in range(numOfLayers):
                print(i)
                if(i == 0):
                    z = np.dot(weights[i], a[i]).T
                elif i == 1:
                    z = np.dot(weights[i].T, a[i].T).T
                else:
                    z = np.dot(weights[i], a[i].T).T
                a.append(sigmoid(z))

                ### Backward propagation
                ybin = np.zeros(a[-1].shape)
                ybin[k][y[r[k]]] = 1  # Assigning 1 to the binary digit according to
                # the class (label) of the input
                #cost += 1 / num_of_training_examples * (-1) * (np.dot(ybin.T, np.log(a[-1])) + np.dot((1 - ybin).T, np.log(1 - a[-1])))
                #deltas[-1] = (a[-1] - ybin)

                g_tag = a[i] * (1 - a[i])
                #deltas[i] = np.dot(weights[i][:, 1:].T, deltas[i]) * g_tag[1:]
                if (i == 0):
                    weightsGrad[i] += np.dot(deltas[i], a[i])
                elif i == 1:
                    weightsGrad[i] += np.dot(deltas[i].T, a[i].T).T
                else:
                    weightsGrad[i] += np.dot(deltas[i], a[i].T)



                weightsDer[i] = 1 / num_of_training_examples * weightsGrad[i]
                weightsDer[i][1:, :] += Lambda / num_of_training_examples * weights[i][1:, :]

                #### Updating the parameters
                weights[i] = weights[i] - learningRate * weightsDer[i]

                cost += (Lambda / (2 * num_of_training_examples)) * np.sum(weights[i] ** 2)

            if np.mod(iter, 2) == 0:
                print('Cost function cost = ', cost, 'in iteration',
                      iter, 'with Lambda = ', Lambda)
                p, acc = ff_predict(weights, X, y)
                print('Net accuracy for training set = ', acc)

    return cost, weights


from keras.datasets import mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()


# for i in range(10):
#     plt.figure(1, figsize = (10,5))
#     plt.imshow(train_images[i])
#     plt.suptitle('label =' + str(train_labels[i]))
#     plt.show()
#     plt.pause(0.1)
    
print('train_images.shape =', train_images.shape)
print('train_labels.shape =', train_labels.shape)
print('test_images.shape =', test_images.shape)
print('test_labels.shape =', test_labels.shape)

# pre-processing
train_images = train_images.reshape((60000, 28*28))
train_images = train_images.astype('float32') / 255

test_images = test_images.reshape((10000, 28*28))
test_images = test_images.astype('float32') / 255

#from keras.utils import to_categorical

# train_labels = to_categorical(train_labels)
# test_labels = to_categorical(test_labels)

X = train_images[:7200,:]
y = train_labels[:7200]
y = y.reshape((y.shape[0], 1))


L1 = X.shape[1]
num_of_units_on_hidden_layers = 16
num_output_units = np.unique(y).size
numOfLayers = int(input())
weights = []
weights.append(init_parameters(L1, num_of_units_on_hidden_layers))
for i in range(1, numOfLayers-1):
    weights.append(init_parameters(num_of_units_on_hidden_layers, num_of_units_on_hidden_layers))
weights.append(init_parameters(num_of_units_on_hidden_layers, num_output_units))

J,weights = backprop(weights, numOfLayers,  X, y, 120, 0.9, 0.01 )

Xtest = test_images[:1000,:]
ytest = test_labels[:1000]
ytest = ytest.reshape((ytest.shape[0], 1))
p, acc = ff_predict(weights, Xtest, ytest)
print('Net accuracy for test set = ', acc)

        
            
        
    












