#!/usr/bin/python3
########################################################################

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d, CubicSpline
import math
from math import tanh, cosh, sin
import h_my_functions_lib as mfl

# some data
x = np.array([0, 1, 2, 3, 4, 5])
y = np.array([0, 2, 1, 3, 7, 8])  

# Define fine-grained x-values for interpolation
x_domain = np.linspace(min(x), max(x), 100)

# Linear Interpolation
linear_interp = interp1d(x, y, kind='linear')
y_linear = linear_interp(x_domain)

# Cubic Spline Interpolation
cubic_spline = CubicSpline(x, y)
y_cubic = cubic_spline(x_domain)

quadratic_spline = interp1d(x, y, kind='quadratic')
y_quadratic = quadratic_spline(x_domain)

# Plot the results
plt.figure(figsize=(8, 5))
plt.scatter(x, y, color='red', label='Data Points', zorder=3)
plt.plot(x_domain, y_linear, '--', label='Linear Interpolation', linewidth=2)
plt.plot(x_domain, y_cubic, label='Cubic Spline Interpolation', linewidth=2)
plt.plot(x_domain, y_quadratic, label='Quadratic Spline Interpolation', linewidth=2)
plt.legend()
plt.xlabel('x')
plt.ylabel('y')
plt.title('Linear vs. Cubic vs. Quadratic Spline Interpolation')
plt.grid(True)
plt.show()

# For sinusoidal data:

# some data
x = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
y = np.sin(x)  

# Define fine-grained x-values for interpolation
x_domain = np.linspace(min(x), max(x), 100)

# Linear Interpolation
linear_interp = interp1d(x, y, kind='linear')
y_linear = linear_interp(x_domain)

# Cubic Spline Interpolation
cubic_spline = CubicSpline(x, y)
y_cubic = cubic_spline(x_domain)

quadratic_spline = interp1d(x, y, kind='quadratic')
y_quadratic = quadratic_spline(x_domain)

# Plot the results
plt.figure(figsize=(8, 5))
plt.scatter(x, y, color='red', label='Data Points', zorder=3)
plt.plot(x_domain, y_linear, '--', label='Linear Interpolation', linewidth=2)
plt.plot(x_domain, y_cubic, label='Cubic Spline Interpolation', linewidth=2)
plt.plot(x_domain, y_quadratic, label='Quadratic Spline Interpolation', linewidth=2)
plt.legend()
plt.xlabel('x')
plt.ylabel('y')
plt.title('Linear vs. Cubic vs. Quadratic Spline Interpolation')
plt.grid(True)
plt.show()

#########################################################################

# Question 1: The cubic splice interpolation is definitely a lot smoother than the linear interpolation. This makes sense because the linear one can only produce a line between each data point.

# Question 2: The linear splice improves a decent amount when doubling the points, regardless of the shape of the data
#             The cubic splice imporves for the original data points, does not seem to get more accurate when doubling the points for a sinusoidal function.

# Question 3: The cubic splice approximates the function better. This makes sense because the sin fuction has a lot of curves and a linear fit can't approximate those very easily

# Question 4: For the original funcion, quadratic seems to be a much better fit. There is very little difference between quadratic and cubic on the sinusoidal function.
#             Due to this small difference, it is not really possible to quickly determine which is better.

#########################################################################


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
h = 10**-8

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
#plt.savefig('numerical_vs_analytic_derivatives.png')
#plt.close()
plt.show()

#########################

# Different h values:

h1 = 2
h2 = 1
h3 = 10**-8
h4 = 10**-16

xdata = np.linspace(x_lower_bound, x_upper_bound, N_samples)

central_diff_values1 = []
for x in xdata:
	central_difference1 = ( mfl.f(x + 0.5*h1) - mfl.f(x - 0.5*h1) ) / h1
	central_diff_values1.append(central_difference1)

central_diff_values2 = []
for x in xdata:
	central_difference2 = ( mfl.f(x + 0.5*h2) - mfl.f(x - 0.5*h2) ) / h2
	central_diff_values2.append(central_difference2)

central_diff_values3 = []
for x in xdata:
	central_difference3 = ( mfl.f(x + 0.5*h3) - mfl.f(x - 0.5*h3) ) / h3
	central_diff_values3.append(central_difference3)

central_diff_values4 = []
for x in xdata:
	central_difference4 = ( mfl.f(x + 0.5*h4) - mfl.f(x - 0.5*h4) ) / h4
	central_diff_values4.append(central_difference4)
## Add the analytical curve
## let's use the same xdata array we already made for our x values

analytical_values1 = []
for x in xdata:
	dfdx1 = mfl.df_dx_analytical(x)
	analytical_values1.append(dfdx1)


#plt.plot(xdata, analytical_values1, linestyle='-', color='black')
#plt.plot(xdata, analytical_values2, linestyle='-', color='blue')
plt.plot(xdata, central_diff_values1, color="green", markersize=8, alpha=0.5)
plt.plot(xdata, central_diff_values2, color="blue", markersize=8, alpha=0.5)
plt.plot(xdata, central_diff_values3, color="red", markersize=8, alpha=0.5)
plt.plot(xdata, central_diff_values4, color="orange", markersize=8, alpha=0.5)
plt.savefig('numerical_vs_analytic_derivatives.png')
#plt.close()
plt.show()


#########################################################################

# Question 5: We provde that sqrt(C) is usually a good value for h, where C is the accuracy. For python this is 10**-16.
#             This means that a good guess for the h value is 10**-8.

# Question 6: 

h1 = 2
h2 = 1
h3 = 10**-8
h4 = 10**-13

xdata = np.linspace(x_lower_bound, x_upper_bound, N_samples)

central_diff_values1 = []
for x in xdata:
	central_difference1 = ( mfl.f(x + 0.5*h1) - mfl.f(x - 0.5*h1) ) / h1
	central_diff_values1.append(central_difference1)

central_diff_values2 = []
for x in xdata:
	central_difference2 = ( mfl.f(x + 0.5*h2) - mfl.f(x - 0.5*h2) ) / h2
	central_diff_values2.append(central_difference2)

central_diff_values3 = []
for x in xdata:
	central_difference3 = ( mfl.f(x + 0.5*h3) - mfl.f(x - 0.5*h3) ) / h3
	central_diff_values3.append(central_difference3)

central_diff_values4 = []
for x in xdata:
	central_difference4 = ( mfl.f(x + 0.5*h4) - mfl.f(x - 0.5*h4) ) / h4
	central_diff_values4.append(central_difference4)
## Add the analytical curve
## let's use the same xdata array we already made for our x values

analytical_values = []
for x in xdata:
	dfdx = mfl.df_dx_analytical(x)
	analytical_values.append(dfdx)


#plt.plot(xdata, analytical_values, linestyle='-', color='black')
plt.plot(xdata, central_diff_values1, color="green", markersize=8, alpha=0.5, label="h=2")
plt.plot(xdata, central_diff_values2, color="blue", markersize=8, alpha=0.5, label="h=1")
plt.plot(xdata, central_diff_values3, color="red", markersize=8, alpha=0.5, label="h^10^-8")
plt.plot(xdata, central_diff_values4, color="orange", markersize=8, alpha=0.5, label="h=10^-13")
plt.savefig('numerical_vs_analytic_derivatives.png')
#plt.close()
plt.legend()
plt.show()


# Question 7: The largest value of h that still provides a good approximation to the analytical derivative is 10^-13
# At 10^-14 the fit starts to get wavy, at 10^-16 it gets very weird.