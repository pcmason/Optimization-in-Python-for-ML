#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 23 21:24:50 2023

Worked example code training a Multilayer Perceptron model using stochastic hill climbing

@author: paulmason
"""

import numpy as np
import math
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


##Transfer function
def transfer(activation):
    #Sigmoid transfer method
    return 1.0 / (1.0 + math.exp(-activation))


##Method that calculates the activation of the model for a given input row of data from dataset
def activate(row, weights):
    #Add the bias, the last weight
    activation = weights[-1]
    #Add the weighted input
    for i in range(len(row)):
        activation += weights[i] * row[i]
    return activation


##Activation function for the network
def predict_row(row, network):
    inputs = row
    #Enumerate the layers in the network from input to output
    for layer in network:
        new_inputs = list()
        #Enumerate nodes in the layer
        for node in layer:
            #Activate the node
            activation = activate(inputs, node)
            #Transfer activation
            output = transfer(activation)
            #Store output
            new_inputs.append(output)
        #Output from this layer is input to the next layer
        inputs = new_inputs
    return inputs[0]


##Use model weights to generate predictions for a dataset of rows
def predict_dataset(x, weights):
    yhats = list()
    for row in x:
        yhat = predict_row(row, weights)
        yhats.append(yhat)
    return yhats


##Objective function
def objective(x, y, network):
    #Generate predictions for dataset
    yhat = predict_dataset(x, network)
    #Round the predictions
    yhat = [round(y) for y in yhat]
    #Calculate the accuracy
    score = accuracy_score(y, yhat)
    return score


##Now get the baseline production of the MLP
#Define a random dataset for binary classification
x, y = make_classification(n_samples = 1000, n_features = 5, n_informative = 2, n_redundant = 1, random_state = 1)
#Determine the number of inputs
n_inputs = x.shape[1]
#One hidden layer and an output layer for the MLP
n_hidden = 10
hidden1 = [np.random.rand(n_inputs + 1) for _ in range(n_hidden)]
output1 = [np.random.rand(n_hidden + 1)]
network = [hidden1, output1]
#Generate predictions for dataset
score = objective(x, y, network)
print(score)


##Step function modifies each weight in the network while simultaneously making a new copy of the network
#Take a step in the search space
def step(network, step_size):
    new_net = list()
    #Enumerate layers in the network
    for layer in network:
        new_layer = list()
        #Enumerate nodes in this layer
        for node in layer:
            #Mutate the node
            new_node = node.copy() + np.random.randn(len(node)) * step_size
            #Store node in layer
            new_layer.append(new_node)
        #Store layer in network
        new_net.append(new_layer)
    return new_net


##Hil climbing local search algorithm
def hillclimbing(x, y, objective, solution, n_iter, step_size):
    #Evalute the initial point
    solution_eval = objective(x, y, solution)
    #Run the hill climb
    for i in range(n_iter):
        #Take a step
        candidate = step(solution, step_size)
        #Evaluate candidate point
        candidate_eval = objective(x, y, candidate)
        #Check if new point should be kept
        if candidate_eval >= solution_eval:
            #Store the new point
            solution, solution_eval = candidate, candidate_eval
            #Report the progress
            print('>%d %f' % (i, solution_eval))
    return [solution, solution_eval]


##Optimize the MLP network using the hillclimbing method
#Split data into train and test datasets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.33)
#Define the total number of iteration
n_iter = 1000
#Define the maximum step size
step_size = 0.1
#Perform the hill climbing search
network, score = hillclimbing(x_train, y_train, objective, network, n_iter, step_size)
print('Done!')
print('Best: %f' % (score))
#Generate predictions for the test dataset
score = objective(x_test, y_test, network)
#output accuracy
print('Test Accuracy: %.5f' % (score * 100))

            