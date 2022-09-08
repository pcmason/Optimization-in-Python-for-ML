#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  7 17:32:07 2022

@author: paulmason
"""
#Worked example of local and global search in Python using SciPy

import numpy as np

#Solve a 2D convex function using L-BFGS-B local search algorithm
from scipy.optimize import minimize

#Objective function
def objective(x):
    return x[0]**2.0 + x[1]**2.0

#Define range of input
r_min, r_max = -5.0, 5.0

#Define starting point as a random sample from the domain
pt = r_min = np.random.rand(2) * (r_max - r_min)

#Perform L-BFGS-B local search 
result = minimize(objective, pt, method = 'L-BFGS-B')

#Summarize the result
print('Local Optimization Results:\n')
print('Status:: %s' % result['message'])
print('Total Evaluations: %d' % result['nfev'])

#Evaluate solution
solution = result['x']
evaluation = objective(solution)
print('Solution: f(%s) = %.5f' % (solution, evaluation))

#Simulated annealing global optimization for a multimodal objective function
from scipy.optimize import dual_annealing

#Objective function 
def objective2(v):
    x, y = v
    return (x**2 + y - 11)**2 + (x + y**2 - 7)**2

#Define the bounds on the search using range of input defined above
bounds = [[-5.0, 5.0], [-5.0, 5.0]]

#Perform simulated annealing search
result = dual_annealing(objective, bounds)

#Summarize the result
print('\nGlobal Optimization Results:\n')
print('Status: %s' % result['message'])
print('Total Evaluations: %d' % result['nfev'])

#Evaluate solution
solution = result['x']
evaluation = objective2(solution)
print('Solution: f(%s) = %5f' % (solution, evaluation))