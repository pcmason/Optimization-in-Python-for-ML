#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 20 00:17:03 2023

Worked example code of hyperparameter optimization on the xgboost algorithm using stochastic hill climbing

@author: paulmason
"""
import numpy as np
import sklearn as skl 
from sklearn.datasets import make_classification
from xgboost import XGBClassifier

##Get the default performance of the XGBoost algorithm
#Define the dataset
x, y = make_classification(n_samples = 1000, n_features = 5, n_informative = 2, n_redundant = 1, random_state = 1)
#Define the model
model = XGBClassifier(verbosity = 0)
#Define evaluation procedure
cv = skl.model_selection.RepeatedStratifiedKFold(n_splits = 10, n_repeats = 3, random_state = 1)
#Evaluate the model
scores = skl.model_selection.cross_val_score(model, x, y, scoring = 'accuracy', cv = cv, n_jobs = -1)
#Report the default accuracy mean and std
print('Mean Accuracy: %.3f (%.3f)' % (np.mean(scores), np.std(scores)))


##Define the objective function
def objective(x, y, cfg):
    #Unpack config
    lrate, n_tree, subsam, depth = cfg
    #Define model
    model = XGBClassifier(learning_rate = lrate, n_estimator = n_tree, subsample = subsam, max_depth = depth, verbosity = 0, use_label_encoder = False)
    #Define evaluation procedure
    cv = skl.model_selection.RepeatedStratifiedKFold(n_splits = 10, n_repeats = 3, random_state = 1)
    #Evaluate the model
    scores = skl.model_selection.cross_val_score(model, x, y, scoring = 'accuracy', cv = cv, n_jobs = -1)
    #Calculate the mean classification accuracy
    result = np.mean(scores)
    return result


##Create method to take a step in the search space
def step(cfg):
    #Unpack config
    lrate, n_tree, subsam, depth = cfg
    
    #Learning rate
    lrate = lrate + np.random.randn() * 0.01
    if lrate <= 0.0:
        lrate = 1e-8
    if lrate > 1:
        lrate = 1.0
    
    #Number of trees
    n_tree = round(n_tree + np.random.randn() * 50)
    if n_tree <= 0.0:
        n_tree = 1
        
    #Subsample percentage
    subsam = subsam + np.random.randn() * 0.1
    if subsam <= 0.0:
        subsam = 1e-8
    if subsam > 1:
        subsam = 1.0
        
    #Max tree depth
    depth = round(depth + np.random.randn() * 7)
    if depth <= 1:
        depth = 1
        
    #Return the new config
    return [lrate, n_tree, subsam, depth]


##Define the local search stochastic hill climbing method
def hillclimbing(x, y, objective, n_iter):
    #Default hyperparameter starting point
    solution = step([0.1, 100, 1.0, 7])
    #Evaluate the initial attempt
    solution_eval = objective(x, y, solution)
    
    #Run the hill climb
    for i in range(n_iter):
        #Take a step
        candidate = step(solution)
        #Evaluate candidate point
        candidate_eval = objective(x, y, candidate)
        
        #Check if new point should be kept
        if candidate_eval >= solution_eval:
            #Store the new point
            solution, solution_eval = candidate, candidate_eval
            #Report progress
            print('>%d, cfg = [%s] %.5f' % (i, solution, solution_eval))
            
    return [solution, solution_eval]


##Optimize the hyperparameters for the XGBoost method
#Define total number of iterations
n_iter = 200
#Perform the hill climbing search
cfg, score = hillclimbing(x, y, objective, n_iter)
print('Done!')
print('cfg = [%s]: Mean Accuracy: %f' % (cfg, score))
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    