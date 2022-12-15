#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 14 22:25:41 2022

@author: paulmason
"""
#Worked Example code using a Gradient Descent method to find an objective function's minimum value

#Import necessary libraries
import numpy as np
from matplotlib import pyplot

#Make the objective method a quadratic function
def objective(x):
    return x**2.0

#Method that calculates derivative of the objective function
def derivative(x):
    return x * 2.0

#Algorithm to calculate gradient descent
def gradient_descent(objective, derivative, bounds, n_iter, step_size):
    #Track all solutions
    solutions, scores = list(), list()
    #Generate an initial point
    solution = bounds[:, 0] + np.random.rand(len(bounds)) * (bounds[:, 1] - bounds[:, 0])
    
    #Run the gradient descent
    for i in range(n_iter):
        #Calculate gradient
        gradient = derivative(solution)
        #Take a step
        solution = solution - step_size * gradient
        #Evaluate candidate point
        solution_eval = objective(solution)
        #Store solution
        solutions.append(solution)
        scores.append(solution_eval)
        #Report progress
        print('>%d f(%s) = %.5f' % (i, solution, solution_eval))
        
    return [solutions, scores]

#Define range of input between -1 and 1
bounds = np.asarray([[-1.0, 1.0]])
#Define total number of iterations
n_iter = 30

#Here uncommment different step sizes to see how it impacts the algorithm
#Define the normal step size
step_size = 0.1
#Really large step size
#step_size = 1
#Really small step size
#step_size = 1e-5


#Perform the gradient descent search
solutions, scores = gradient_descent(objective, derivative, bounds, n_iter, step_size)
#Sample input range uniformly at 0.1 increments
inputs = np.arange(bounds[0,0], bounds[0,1] + 0.1, 0.1)
#Compute targets
results = objective(inputs)
#Create a line plot comparing input vs result
pyplot.plot(inputs, results)
#Plot the solutions found
pyplot.plot(solutions, scores, '.-', color = 'red')
pyplot.show()




