#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 19 22:50:49 2023

Worked example code of hyperparameter optimization for the perceptron algorithm using stochastic hill climbing


@author: paulmason
"""

from sklearn.datasets import make_classification
from sklearn.linear_model import Perceptron
import numpy as np
import sklearn as skl

##Evaluate perceptron model with default hyperparameters
#Define a binary classification dataset
x, y = make_classification(n_samples = 1000, n_features = 5, n_informative = 2, n_redundant = 1, random_state = 1)
#Summarize the shape of the dataset
print(x.shape, y.shape)
#Define the model
model = Perceptron()
#Define evaluation procedure
cv = skl.model_selection.RepeatedStratifiedKFold(n_splits = 10, n_repeats = 3, random_state = 1)
#Evaluate model
scores = skl.model_selection.cross_val_score(model, x, y, scoring = 'accuracy', cv = cv, n_jobs = -1)
#Report result
print('Mean Accuracy: %.3f (%.3f)' % (np.mean(scores), np.std(scores)))

##Define an objective function that is evaluated with mean classification accuracy
##with repeated stratified k-fold cross-validation
def objective(x, y, cfg):
    #Unpack config
    eta, alpha = cfg
    #Define the model
    model = Perceptron(penalty = 'elasticnet', alpha = alpha, eta0 = eta)
    #Define evaluation procedure
    scores = skl.model_selection.cross_val_score(model, x, y, scoring = 'accuracy', cv = cv, n_jobs = -1)
    #Calculate the mean accuracy
    result = np.mean(scores)
    return result

##Function to take a step in the search space
def step(cfg, step_size):
    #Unpack config
    eta, alpha = cfg
    
    #Step eta
    new_eta = eta + np.random.randn() * step_size  
    #Check the bounds of eta
    if new_eta <= 0.0:
        new_eta = 1e-8
        
    #Step alpha
    new_alpha = alpha + np.random.randn() * step_size
    #Check the bounds of alpha
    if new_alpha < 0.0:
        new_alpha = 0.0
        
    #Return the new config
    return [new_eta, new_alpha]

##Hill climbing local search algorithm
def hillclimbing(x, y, objective, n_iter, step_size):
    #Get random starting point for the search
    solution = [np.random.rand(), np.random.rand()]
    #Evaluate the initial point
    solution_eval = objective(x, y, solution)

    #Run the hill climb
    for i in range(n_iter):
        #Take a step
        candidate = step(solution, step_size)
        #Evaluate candidate point
        candidate_eval = objective(x, y, candidate)
        
        #Check if new point should be kept
        if candidate_eval >= solution_eval:
            #Store the point
            solution, solution_eval = candidate, candidate_eval
            #Report progress
            print('>%d, cfg = %s %.5f' % (i, solution, solution_eval))
    
    return [solution, solution_eval]

##Call the hillclimbing algorithm and report the results of the search
#Define total number of iterations
n_iter = 100
#Step size in the search space
step_size = 0.1
#Perform the hill climbing search
cfg, score = hillclimbing(x, y, objective, n_iter, step_size)
print('Done!')
print('cfg = %s: Mean Accuracy: %f' % (cfg, score))
