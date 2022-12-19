#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 17 20:52:41 2022

@author: paulmason
"""
#Worked example of gradient descent optimized with adam

#Import libs
import math
import numpy as np
from matplotlib import pyplot as plt

#Create 2D multivariate objective function
def objective(x, y):
    return x ** 2.0 + y ** 2.0

#Derivative function for objective method
def derivative(x, y):
    return np.asarray([x * 2.0, y * 2.0])

#Gradient descent with adam
def adam(objective, derivative, bounds, n_iter, alpha, beta1, beta2, eps = 1e-8):
    #Track solutions to be able to plot progress later
    solutions = list()
    #Generate initial point
    x = bounds[:, 0] + np.random.rand(len(bounds)) * (bounds[:, 1] - bounds[:, 0])
    score = objective(x[0], x[1])
    #Initialize first and second moments to 0
    m = [0.0 for _ in range(bounds.shape[0])]
    v = [0.0 for _ in range(bounds.shape[0])]
    
    #Run the gradient descent updates
    for t in range(n_iter):
        #Calculate gradient g(t)
        g = derivative(x[0], x[1])
        
        #Build a solution one variable at a time
        for i in range(x.shape[0]):
            #Next first moment
            m[i] = beta1 * m[i] + (1.0 - beta1) * g[i]
            #Next second moment
            v[i] = beta2 * v[i] + (1.0 - beta2) * g[i]**2.0
            #Adjust m for the bias since initialized to 0
            mhat = m[i] / (1.0 - beta1**(t+1))
            #Adjust v for the bias since initialized to 0
            vhat = v[i] / (1.0 - beta2**(t+1))
            #Now get the next initial point in the gradient descent algorithm
            x[i] = x[i] - alpha * mhat / (math.sqrt(vhat) + eps)
            #Evaluate candidate point
            score = objective(x[0], x[1])
            #Keep track of solutions
            solutions.append(x.copy())
            
        #Report progress
        print('>%d f(%s) = %.5f' % (t, x, score))
        
    return [x, score, solutions]

##Create constants that will be used throughout code below
#Define range for input
R_MIN, R_MAX = -1.0, 1.0
BOUNDS = np.asarray([[R_MIN, R_MAX], [R_MIN, R_MAX]])
#Sample input range uniformly using 0.1 increments
XAXIS = np.arange(R_MIN, R_MAX, 0.1)
YAXIS = np.arange(R_MIN, R_MAX, 0.1)
#Create a mesh from the axis
X, Y = np.meshgrid(XAXIS, YAXIS)
#Compute targets
RESULTS = objective(X, Y)


##Create 3D Plot of Dataset
#Create surface plot with jet color scheme
figure = plt.figure()
axis = figure.gca(projection = '3d')
axis.plot_surface(X, Y, RESULTS, cmap = 'jet')
plt.show()

##Create 2D Plot of the Function
#Create a filled contour plot with 50 levels and jet color scheme
plt.contourf(X, Y, RESULTS, levels = 50, cmap = 'jet')
plt.show()

##Call the adam method to optimize the objective function
#Seed the RNG
np.random.seed(1)
#Define total # of iterations
n_iter = 60
#Step size
alpha = 0.02
#Factor for average gradient
beta1 = 0.8
#Factor for average squared gradient
beta2 = 0.999
#Perform the gradient descent search with adam and output the results
best, score, solutions = adam(objective, derivative, BOUNDS, n_iter, alpha, beta1, beta2)
print('Done!')
print('f(%s) = %f' % (best, score))

##Create contour plot of objective function and plot progress of the adam GD
#Create a filled contour plot with 50 levels and jet color scheme
plt.contourf(X, Y, RESULTS ,levels = 50, cmap = 'jet')
#Plot the sample as white circles
solutions = np.asarray(solutions)
plt.plot(solutions[:, 0], solutions[:, 1], '.-', color = 'w')
plt.show()