#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan  2 19:47:40 2023

@author: paulmason
"""
#Worked example of random search for function optimization

import numpy as np
from matplotlib import pyplot

#Basic objective function
def objective(x):
    return x**2.0

#Define range for the input
r_min, r_max = -5.0, 5.0
#Generate 100 random samples from the domain
sample = r_min + np.random.rand(100) * (r_max - r_min)
#Evaluate the sample
sample_eval = objective(sample)
#Locate the best solution
best = 0
for i in range(len(sample)):
    if sample_eval[i] < sample_eval[best]:
        best = i
        
#Summarize best solution
print('Best: f(%.5f) = %.5f' % (sample[best], sample_eval[best]))

#Sample input range uniformly at 0.1 increments
inputs = np.arange(r_min, r_max, 0.1)
#Compute targets
results = objective(inputs)
#Create a line plot of input vs result
pyplot.plot(inputs, results)
#Plot the sample
pyplot.scatter(sample, sample_eval)
#Draw a vertical line at the best input
pyplot.axvline(x = sample[best], ls = '--', color = 'red')
pyplot.show()