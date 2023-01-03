#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan  2 20:31:16 2023

@author: paulmason
"""
#Worked example of grid search for function optimization

import numpy as np
from matplotlib import pyplot

#Define simple 2D objective function
def objective(x, y):
    return x**2.0 + y**2.0

#Define range for input
r_min, r_max = -5.0, 5.0
#Generate a grid sample from the domain
sample = list()
step = 0.5
for x in np.arange(r_min, r_max + step, step):
    for y in np.arange(r_min, r_max + step, step):
        sample.append([x, y])
        
#Evaluate the sample
sample_eval = [objective(x, y) for x,y in sample]
#Locate the best solution
best = 0
for i in range(len(sample)):
    if sample_eval[i] < sample_eval[best]:
        best = i
        
#Summarize best solution
print('Best: f(%.5f, %.5f) = %.5f' % (sample[best][0], sample[best][1], sample_eval[best]))

#Sample input range uniformly at 0.1 increments
xaxis = np.arange(r_min, r_max, 0.1)
yaxis = np.arange(r_min, r_max, 0.1)
#Create a mesh from the axis
x, y = np.meshgrid(xaxis, yaxis)
#Compute targets
results = objective(x, y)
#Create a filled contour plot
pyplot.contourf(x, y, results, levels = 50, cmap = 'jet')
#Plot the sample as black circles
pyplot.plot([x for x, _ in sample], [y for _, y in sample], '.', color = 'black')
#Draw the best result as a white star
pyplot.plot(sample[best][0], sample[best][1], '*', color = 'white')
pyplot.show()