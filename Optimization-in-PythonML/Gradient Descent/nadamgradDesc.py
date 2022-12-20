#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 19 18:44:21 2022

@author: paulmason
"""
#Worked example code of gradient descent optimized with nadam

#Import libs
import numpy as np
from matplotlib import pyplot as plt
import math

#Create multi-dimensional objective function
def objective(x, y):
    return x**2.0 + y**2.0

#Derivative of objective function
def derivative(x, y):
    return np.asarray([x * 2.0,y * 2.0])

#Gradient descent algorithm with nadam
def nadam(objective, derivative, bounds, n_iter, alpha, mu, nu, eps = 1e-8):
    #List to keep track of solutions
    solutions = list()
    #Generate initial point
    x = bounds[:, 0] + np.random.rand(len(bounds)) * (bounds[:, 1] - bounds[:, 0])
    score = objective(x[0], x[1])
    #Initialize decaying moving averages
    m = [0.0 for _ in range(bounds.shape[0])]
    n = [0.0 for _ in range(bounds.shape[0])]
    #Run the gradient descent
    
    for t in range(n_iter):
        #Calculate gradient g(t)
        g = derivative(x[0], x[1])
        #Build a solution 1 variable at a time
        
        for i in range(bounds.shape[0]):
            #Update first moment
            m[i] = mu * m[i] + (1.0 - mu) * g[i]
            #Update second moment
            n[i] = nu * n[i] + (1.0 - nu) * g[i]**2
            #Bias correct first and second moment
            mhat = (mu * m[i] / (1.0 - mu)) + ((1.0 - mu) * g[i] / (1.0 - mu))
            nhat = nu * n[i] / (1.0 - nu)
            #Get the point for this iteration
            x[i] = x[i] - alpha / (math.sqrt(nhat) + eps) * mhat
            
        #Evaluate candidate point
        score = objective(x[0], x[1])
        #Store solution
        solutions.append(x.copy())
        #Report progress 
        print('>%d f(%s) = %.5f' % (t, x, score))
        
    return [x, score, solutions]

##Constants that are used throughout program
#Range for input
R_MIN, R_MAX = -1.0, 1.0
BOUNDS = np.asarray([[R_MIN, R_MAX], [R_MIN, R_MAX]])
#Sample input range uniformly at 0.1 increments
XAXIS = np.arange(R_MIN, R_MAX, 0.1)
YAXIS = np.arange(R_MIN, R_MAX, 0.1)
#Create a mesh from the axis
X, Y = np.meshgrid(XAXIS, YAXIS)
#Compute targets
RESULTS = objective(X, Y)

##Create 3D plot of dataset
#Create surface plot with jet color scheme
figure = plt.figure()
axis = figure.gca(projection = '3d')
axis.plot_surface(X, Y, RESULTS, cmap = 'jet')
plt.show()

##Create 2D plot of dataset
#Create a filled contour plot with 50 levels and jet color scheme
plt.contourf(X, Y, RESULTS, levels = 50, cmap = 'jet')
plt.show()

##Run nadam on the objective function
#Seed the RNG
np.random.seed(1)
#Define total # of iterations
n_iter = 50
#Step size
alpha = 0.2
#Factor for average gradient
mu = 0.8
#Factor for average squared gradient
nu = 0.999
#Perform gradient descent search with nadam
best, score, solutions = nadam(objective, derivative, BOUNDS, n_iter, alpha, mu, nu)
#Summarize result
print('Done!')
print('f(%s) = %f' %(best, score))

##Track the nadam function
#Create a filled contour plot with 50 levels and jet color scheme
plt.contourf(X, Y, RESULTS, levels = 50, cmap = 'jet')
#Plot the sample as white circles
solutions = np.asarray(solutions)
plt.plot(solutions[:, 0], solutions[:, 1], '.-', color = 'w')
plt.show()