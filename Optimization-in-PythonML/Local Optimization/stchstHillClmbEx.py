#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 21 21:15:23 2022

@author: paulmason
"""
import numpy as np
from matplotlib import pyplot
#Define x^2 objective function
def objective(x):
    return x[0]**2.0

#Hill climbing local search algorithm
def hillclimbing(objective, bounds, n_iterations, step_size):
    #Generate an initial point
    solution = bounds[:, 0] + np.random.rand(len(bounds)) * (bounds[:, 1] - bounds[:, 0])
    #Evaluate the initial point
    solution_eval = objective(solution)
    #Add the score to a total scores list
    scores = list()
    scores.append(solution_eval)
    #Run the hill climb
    for i in range(n_iterations):
        #Take a step
        candidate = solution + np.random.randn(len(bounds)) * step_size
        #Evaluate candidate point
        candidate_eval = objective(candidate)
        #Check if the new point should replace old point
        if candidate_eval <= solution_eval:
            #Store the new point
            solution, solution_eval = candidate, candidate_eval
            #Keep track of scores
            scores.append(solution_eval)
            #Report progress
            print('>%d f(%s) = %5f' % (i, solution, solution_eval))
    
    return [solution, solution_eval, scores]

#Seed the pseudorandom number generator
np.random.seed(5)
#Define range for input
bounds = np.asarray([[-5.0, 5.0]])
#Define range for input
n_iterations = 1000
#Define the max step size
step_size = 0.1

#Perform the hill climbing search
best, score, scores = hillclimbing(objective, bounds, n_iterations, step_size)
print('Done!')
print('f(%s) = %f' % (best, score))

#Line plot of best scores
pyplot.plot(scores, '.-')
pyplot.xlabel('Improvement Number')
pyplot.ylabel('Evaluation f(x)')
pyplot.show()

#Sample input range uniformly at 0.1 increments
inputs = np.arange(bounds[0, 0], bounds[0, 1], 0.1)
#Create a line plot of input vs result
pyplot.plot(inputs, [objective([x]) for x in inputs], '--')
#Draw a vertical line at the optimal input
pyplot.axvline(x = [0.0], ls = '--', color = 'red')
#Plot the sample as black circles
pyplot.plot(scores, [objective([x]) for x in scores], 'o', color = 'black')
pyplot.show()