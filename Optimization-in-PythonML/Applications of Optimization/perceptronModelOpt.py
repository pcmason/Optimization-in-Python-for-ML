#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 21 23:17:24 2023

Worked example code training a Perceptron neural network using stochastic hill climbing

@author: paulmason
"""

from sklearn.datasets import make_classification
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

##Create binary classification dataset and summarize its shape
#Define dataset with 1000 rows and 5 input variables
x, y = make_classification(n_samples = 1000, n_features = 5, n_informative = 2, n_redundant = 1, random_state = 1)
#Summarize the shape of the dataset
print(x.shape, y.shape)


##Create the step transfer function
def transfer(activation):
    if activation >= 0:
        return 1
    return 0


##Method that calculates the activation of the model for a given input row of data from dataset
def activate(row, weights):
    #Add the bias, the last weight
    activation = weights[-1]
    #Add the weighted input
    for i in range(len(row)):
        activation += weights[i] * row[i]
    return activation


##Use model weights to predict 0/1 for a given row of data
def predict_row(row, weights):
    #Activate for input
    activation = activate(row, weights)
    #Transfer for activation
    return transfer(activation)


##Use model weights to generate predictions for a dataset of rows
def predict_dataset(x, weights):
    yhats = list()
    for row in x:
        yhat = predict_row(row, weights)
        yhats.append(yhat)
    return yhats


##Use perceptron model to make predictions on dataset created above
#Determine the number of weights
n_weights = x.shape[1] + 1
#Generate random weights
weights = np.random.rand(n_weights)
#Generate predictions for dataset
yhat = predict_dataset(x, weights)
#Calculate the accuracy
score = accuracy_score(y, yhat)
print('Initial Score without tuning: %.2f\n' % (score * 100))


##Define an objective function to return the accuracy of the model (used for hillclimbing method)
def objective(x, y, weights):
    #Generate predictions for the dataset
    yhat = predict_dataset(x, weights)
    #Calculate accuracy
    score = accuracy_score(y, yhat)
    return score


##Define the hill climbing local search algorithm
def hillclimbing(x, y, objective, solution, n_iter, step_size):
    #Evaluate the initial point
    solution_eval = objective(x, y, solution)
    #Run the hill climb
    for i in range(n_iter):
        #Take a step
        candidate = solution + np.random.randn(len(solution)) * step_size
        #Evaluate candidate point
        candidate_eval = objective(x, y, candidate)
        #Check if new point should be keep over current best solution
        if candidate_eval >= solution_eval:
            #Store the new point
            solution, solution_eval = candidate, candidate_eval
            #Report the progress
            print('>%d %.5f' % (i, solution_eval))
    return [solution, solution_eval]


##Optimize the weights of the dataset to achieve good accuracy
#Split dataset into train and test sets, test set is 33% of total set
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.33)
#Define total number of iterations
n_iter = 1000
#Define the max step size
step_size = 0.05
#Define the initial solution
solution = np.random.rand(n_weights)
#Perform the hill climbing search
weights, score = hillclimbing(x_train, y_train, objective, solution, n_iter, step_size)
print('Done!')
print('f(%s) = %f' % (weights, score))
#Generate predictions for the test dataset
yhat = predict_dataset(x_test, weights)
#Calculate the accuracy
score = accuracy_score(y_test, yhat)
print('Test Accuracy: %.5f' % (score * 100))

