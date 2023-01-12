#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 11 21:19:33 2023

Worked example code using simulated annealing global optimization method.

@author: paulmason
"""
import numpy as np
from matplotlib import pyplot

#Define 1D objective function
def objective(x):
    return x[0]**2.0

#Simulated annealing algorithm
def simulated_annealing(objective, bounds, n_iterations, step_size, temp):
    #Create list to store best scores in
    scores = list()
    #Generate an initial point
    best = bounds[:, 0] + np.random.rand(len(bounds)) * (bounds[:, 1] - bounds[:, 0])
    #Evaluate initial point
    best_eval = objective(best)
    #Current working solution
    curr, curr_eval = best, best_eval
    #Run the algorithm
    for i in range(n_iterations):
        #Take a step
        candidate = curr + np.random.randn(len(bounds)) * step_size
        #Evaluate candidate point
        candidate_eval = objective(candidate)
        #Check for the new best solution
        if candidate_eval < best_eval:
            #Store new best point
            best, best_eval = candidate, candidate_eval
            #Store the best scores
            scores.append(best_eval)
            #Report progress
            print('>%d f(%s) = %.5f' % (i, best, best_eval))
            #Difference between candidate and currrent point evaluation
            diff = candidate_eval - curr_eval
            #Calculate temperature for current epoch
            t = temp / float(i + 1)
            #Calculate metropolis acceptance criterion
            metropolis = np.exp(-diff / t)
            #Check if we should keep the new point
            if diff < 0 or np.random.rand() < metropolis:
                #Store the new current point
                curr, curr_eval = candidate, candidate_eval
                
    return [best, best_eval, scores]

#Define range for input
r_min, r_max = -5.0, 5.0
bounds = np.asarray([[r_min, r_max]])
#Sample input range uniformly at 0.1 increments
inputs = np.arange(r_min, r_max, 0.1)
#Compute targets
results = [objective([x]) for x in inputs]
#Create a line plot of input vs result
pyplot.plot(inputs, results)
#Define optimal input value
x_optima = 0.0
#Draw a vertical line at the optimal input
pyplot.axvline(x = x_optima, ls = '--', color = 'red')
#Show the plot
pyplot.show()

##Explore temperature vs algorithm iteration for simulated annealing
#total iterations of algorithm
iterations = 100
#Initial temperature
initial_temp = 10
#Array of iterations from 0 to iterations - 1
iterations = [i for i in range(iterations)]
#Temperatures for each iterations
temperatures = [initial_temp / float(i + 1) for i in iterations]
#Metropolis acceptance criterion
differences = [0.01, 0.1, 1.0]
for d in differences:
    metropolis = [np.exp(-d / t) for t in temperatures]
    #Plot iterations vs metropolis
    label = 'diff = %.2f' % d
    pyplot.plot(iterations, metropolis, label = label)
#Initialize plot
pyplot.xlabel('Iteration')
pyplot.ylabel('Metropolis Criterion')
pyplot.legend()
pyplot.show()
#Plot ierations vs temperatures
pyplot.plot(iterations, temperatures)
pyplot.xlabel('Iteration')
pyplot.ylabel('Temperature')
pyplot.show()

##Worked example of simulated annealing algorithm
#Seed the psuedo RNG
np.random.seed(1)
#Number of iterations
n_iterations = 1000
#Maximum step size
step_size = 0.1
#Initial temperature
temp = 10
#Perform the simulated annealing search
best, score, scores = simulated_annealing(objective, bounds, n_iterations, step_size, temp)
print('Done!')
print('f(%s) = %f' % (best, score))

#Line plot of best scores
pyplot.plot(scores, '.-')
pyplot.xlabel('Improvement Number')
pyplot.ylabel('Evaluation f(x)')
pyplot.show()