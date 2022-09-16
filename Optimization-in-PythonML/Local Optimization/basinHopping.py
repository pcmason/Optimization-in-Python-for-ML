#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 15 20:52:50 2022

@author: paulmason
"""
#Worked example of basin hopping in Python with 2 distinct mutltimodal problems and solutions

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.optimize import basinhopping

#First example
#Multimodal optimization with local optima (use Ackley function)
def objective1(x, y):
    return -20.0 * np.exp(-0.2 * np.sqrt(0.5 * (x**2 + y**2))) - np.exp(0.5 * (np.cos(2 * np.pi * x) + np.cos(2 * np.pi * y))) + np.e +20

#Same function as above but with 1 paramater instead of 2
def objective2(v):
    x, y = v
    return -20.0 * np.exp(-0.2 * np.sqrt(0.5 * (x**2 + y**2))) - np.exp(0.5 * (np.cos(2 * np.pi * x) + np.cos(2 * np.pi * y))) + np.e +20

#Define range for input
r_min, r_max = -5.0, 5.0

#Sample input range at 0.1 increments
xaxis = np.arange(r_min, r_max, 0.1)
yaxis = np.arange(r_min, r_max, 0.1)

#Create a mesh from the axis
x, y = np.meshgrid(xaxis, yaxis)

#Compute targets 
results = objective1(x, y)

#Create surface plot with jet color scheme
figure = plt.figure()
axis = figure.gca(projection = '3d')
axis.plot_surface(x, y, results, cmap = 'jet')

#Show the plot
plt.show()

#Define the starting point as a random sample from the domain
pt = r_min = np.random.rand(2) * (r_min - r_max)

#Perform basin hopping search 
result = basinhopping(objective2, pt, stepsize = 0.5, niter = 200)

#Summarize the result
print('First Example:')
print('Status: %s' % result['message'])
print('Total Evaluations: %d' % result['nfev'])

#Evaluate solution
solution = result['x']
evaluation = objective2(solution)
print('Solution: f(%s) = %.5f' % (solution, evaluation))


#Second example
#Multimodal optimization with multiple global optima (Himmelblau function)
def himmelblau1(x, y):
    return (x**2 + y - 11)**2 + (x + y**2 - 7)**2

#Same function as above but with 1 instead of 2 parameters
def himmelblau2(v):
    x, y = v
    return (x**2 + y - 11)**2 + (x + y**2 - 7)**2

#Compute targets
results = himmelblau1(x, y)

#Create surface plot for visualization
figure = plt.figure()
axis = figure.gca(projection = '3d')
axis.plot_surface(x, y, results, cmap = 'jet')
plt.show()

#Perform basin hopping search
result = basinhopping(himmelblau2, pt, stepsize = 0.5, niter = 200)

#Summarize the result
print('\nSecond Example:')
print('Status: %s' % result['message'])
print('Total Evaluations: %d' % result['nfev'])

#Evaluation solution
solution = result['x']
evaluation = himmelblau2(solution)
print('Solution: f(%s) = %.5f' % (solution, evaluation))