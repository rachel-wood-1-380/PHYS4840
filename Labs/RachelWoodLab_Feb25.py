#!/usr/bin/python3
########################################################################


# In-class Exercise A):
#import h_my_functions_lib


import math
def function2(x):
	return 1 + 1/2 * math.tan(2*x)

print(function2(2))

h = 1e-10

for x in [-2, 2, 4/1e-10]:
	centraldiff = ((function2(x + h/2) - function2(x - h/2)) / h)
	print(centraldiff)



import numpy as np
import matplotlib.pyplot as plt
from math  import tanh, cosh

import sys
sys.path.append('../')
import my_functions_lib as mfl

## compute the instantaneous derivatives
## using the central difference approximation
## over the interval -2 to 2

x_lower_bound = -2.0
x_upper_bound = 2.0

N_samples = 100

#####################
#
# Try different values of h
# What did we "prove" h should be
# for C = 10^(-16) in Python?
#
#######################
h = ... ## what goes here?

xdata = np.linspace(x_lower_bound, x_upper_bound, N_samples)

central_diff_values = []
for x in xdata:
	central_difference = ( mfl.f(x + 0.5*h) - mfl.f(x - 0.5*h) ) / h
	central_diff_values.append(central_difference)

## Add the analytical curve
## let's use the same xdata array we already made for our x values

analytical_values = []
for x in xdata:
	dfdx = mfl.df_dx_analytical(x)
	analytical_values.append(dfdx)


plt.plot(xdata, analytical_values, linestyle='-', color='black')
plt.plot(xdata, central_diff_values, "*", color="green", markersize=8, alpha=0.5)
plt.savefig('numerical_vs_analytic_derivatives.png')
plt.close()