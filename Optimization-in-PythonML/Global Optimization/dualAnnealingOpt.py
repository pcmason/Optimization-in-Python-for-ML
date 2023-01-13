#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 12 20:39:02 2023

Worked example code using dual annealing global optimization method.

@author: paulmason
"""
import numpy as np
from matplotlib import pyplot
from scipy.optimize import dual_annealing

#Use Ackerley Function as objective function
def objective(v):
    x, y = v
    return -20.0 * np.exp(-0.2 * np.sqrt(0.5 * (x**2 + y**2))) - np.exp(0.5 * (np.cos(2 * np.pi * x) + np.cos(2 * np.pi * y))) + np.e + 20

##Output graph of Ackerley Function
#Define range for input
r_min, r_max = -5.0, 5.0
#Sample input range uniformly at 0.1 increments
xaxis = np.arange(r_min, r_max, 0.1)
yaxis = np.arange(r_min, r_max, 0.1)
#Create a mesh from the axis
x, y = np.meshgrid(xaxis, yaxis)
#Compute targets
v = x, y
results = objective(v)
#Create a surface plot with the jet color scheme
figure = pyplot.figure()
axis = figure.gca(projection = '3d')
axis.plot_surface(x, y, results, cmap = 'jet')
pyplot.show()

##Apply dual annealing to Ackerley Function
#Define the bounds on the search
bounds = [[r_min, r_max], [r_min, r_max]]
#Perform the dual annealig search
result = dual_annealing(objective, bounds)
#Summarize the result
print('Status : %s' % result['message'])
print('Total Evaluations: %d' % result['nfev'])
#Evaluate the solution
solution = result['x']
evaluation = objective(solution)
print('Solution: f(%s) = %.5f' % (solution, evaluation))