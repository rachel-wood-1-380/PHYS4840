#!/usr/bin/python3
#########################################

import numpy as np
import matplotlib.pyplot as plt
from h_my_functions_lib import euler_method
from h_my_functions_lib import runge_kutta_method

# Which part of the coding function is f(x)?
	# It is the for-loop I think.

# How do we choose step size h?
	# I think that step size h should be equal to dt. We should choose this value by finding
	# the leading error term and using the formula (C^n)^2

# What operation do we want to do in a for-loop when we're integrating?

# Example 8.1 from the textbook:

def f(x, t):
	return -x**3 + np.sin(t)

x0 = 0.0
t0 = 0.0
t_end = 10.0
dt = (10 - 0) / 1000

t_values, x_values = euler_method(f, x0, t0, t_end, dt)

plt.plot(t_values, x_values)
plt.xlabel("t")
plt.ylabel("x(t)")
plt.title("Euler method for -x^**3 + sin(t)")
plt.show()

# In - class example:

# Example 8.2 from the textbook:
N1, N2, N3, N4 = 10, 20, 50, 100
t_values1, x_values1 = runge_kutta_method(f, x0, t0, t_end, N1)
t_values2, x_values2 = runge_kutta_method(f, x0, t0, t_end, N2)
t_values3, x_values3 = runge_kutta_method(f, x0, t0, t_end, N3)
t_values4, x_values4 = runge_kutta_method(f, x0, t0, t_end, N4)

plt.plot(t_values1, x_values1)
plt.show()