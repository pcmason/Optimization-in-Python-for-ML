#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 15 19:44:13 2023

Worked example solving the Ackley Function using Differential Evolution. 

@author: paulmason
"""
import numpy as np
from matplotlib import pyplot
from scipy.optimize import differential_evolution

#Objective function is the Ackley Function
def objective(v):
    x, y = v
    return -20.0 * np.exp(-0.2 * np.sqrt(0.5 * (x**2 + y**2))) - np.exp(0.5 * (np.cos(2 * np.pi * x) + np.cos(2 * np.pi * y))) + np.e + 20

##Create 3D suface plot showing the global optima
#Define range for input
r_min, r_max = -7.0, 7.0
#Sample input range uniformly at 0.1 increments
xaxis = np.arange(r_min, r_max, 0.1)
yaxis = np.arange(r_min, r_max, 0.1)
#Create a mesh from the axis
x, y = np.meshgrid(xaxis, yaxis)
v = x, y
#Compute targets
results = objective(v)
#Create a surface plot with the jet color scheme
figure = pyplot.figure()
axis = figure.gca(projection = '3d')
axis.plot_surface(x, y, results, cmap = 'jet')
pyplot.show()

##Apply differential evolution to Ackley function
#Define bounds on the search
bounds = [[r_min, r_max], [r_min, r_max]]
#Perform the differential evolution search
result = differential_evolution(objective, bounds)
#Summarize the result
print('Status: %s' % result['message'])
print('Total Evaluations: %d' % result['nfev'])
#Evaluate the solution
solution = result['x']
evaluation = objective(solution)
print('Solution: f(%s) = %.5f' % (solution, evaluation))