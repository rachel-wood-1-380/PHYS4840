#!/usr/bin/python3
#########################################

# In-Class Question 1: What happens when you type >$ which gfortran ?
# > /usr/bin/gfortran

# In-Class Exercise 1: Example 8.2 in the textbook.

import math
import numpy as np
import pylab
import matplotlib.pyplot as plt

# Define our function:
def f(x, t):
	return -x**3 + math.sin(t)

# Define our endpoints and stepsize:
a = 0.0
b = 10.0
N1 = 10
N2 = 20
N3 = 50
N4 = 100
h1 = (b - a ) / N1
h2 = (b - a ) / N2
h3 = (b - a ) / N3
h4 = (b - a ) / N4

# Our t values between a and b, spaced h apart.
tpoints1 = np.arange(a, b, h1)
tpoints2 = np.arange(a, b, h2)
tpoints3 = np.arange(a, b, h3)
tpoints4 = np.arange(a, b, h4)
# x points, empty list
xpoints1 = []
xpoints2 = []
xpoints3 = []
xpoints4 = []

# Initial value for x
x = 0.0

# Apply the Runge-Kutta stuff:

for t in tpoints1:
	xpoints1.append(x)
	k1 = h1*f(x, t)
	k2 = h1*f(x + 0.5 * k1,t + 0.5 * h1)
	x += k2

for t in tpoints2:
	xpoints2.append(x)
	k1 = h2*f(x, t)
	k2 = h2*f(x + 0.5 * k1,t + 0.5 * h2)
	x += k2

for t in tpoints3:
	xpoints3.append(x)
	k1 = h3*f(x, t)
	k2 = h3*f(x + 0.5 * k1,t + 0.5 * h3)
	x += k2

for t in tpoints4:
	xpoints4.append(x)
	k1 = h4*f(x, t)
	k2 = h4*f(x + 0.5 * k1,t + 0.5 * h4)
	x += k2


pylab.plot(tpoints1, xpoints1, label="N = 10")
pylab.plot(tpoints2, xpoints2, label="N = 20")
pylab.plot(tpoints3, xpoints3, label="N = 50")
pylab.plot(tpoints4, xpoints4, label="N = 100")
pylab.xlabel("t")
pylab.ylabel("x(t)")
pylab.legend()
pylab.title("Runge-Kutta 2 for -x**3 + sin(t)")
pylab.show()

# In-Class Exercise 2: Example 8.3 in the textbook:

tpoints1 = np.arange(a, b, h1)
tpoints2 = np.arange(a, b, h2)
tpoints3 = np.arange(a, b, h3)
tpoints4 = np.arange(a, b, h4)
# x points, empty list
xpoints1 = []
xpoints2 = []
xpoints3 = []
xpoints4 = []

x = 0.0

for t in tpoints1:
	xpoints1.append(x)
	k1 = h1*f(x, t)
	k2 = h1*f(x + 0.5 * k1,t + 0.5 * h1)
	k3 = h1*f(x + 0.5 * k2, t + 0.5 * h1)
	k4 = h1 * f(x + k3, t + h1)
	x += (k1 + 2 * k2 + 2 * k3 + k4) / 6

for t in tpoints2:
	xpoints2.append(x)
	k1 = h2*f(x, t)
	k2 = h2*f(x + 0.5 * k1,t + 0.5 * h2)
	k3 = h2*f(x + 0.5 * k2, t + 0.5 * h2)
	k4 = h2 * f(x + k3, t + h2)
	x += (k1 + 2 * k2 + 2 * k3 + k4) / 6

for t in tpoints3:
	xpoints3.append(x)
	k1 = h3*f(x, t)
	k2 = h3*f(x + 0.5 * k1,t + 0.5 * h3)
	k3 = h3*f(x + 0.5 * k2, t + 0.5 * h3)
	k4 = h3 * f(x + k3, t + h3)
	x += (k1 + 2 * k2 + 2 * k3 + k4) / 6

for t in tpoints4:
	xpoints4.append(x)
	k1 = h4*f(x, t)
	k2 = h4*f(x + 0.5 * k1,t + 0.5 * h4)
	k3 = h4*f(x + 0.5 * k2, t + 0.5 * h4)
	k4 = h4 * f(x + k4, t + h4)
	x += (k1 + 2 * k2 + 2 * k3 + k4) / 6


pylab.plot(tpoints1, xpoints1, label="N = 10")
pylab.plot(tpoints2, xpoints2, label="N = 20")
pylab.plot(tpoints3, xpoints3, label="N = 50")
pylab.plot(tpoints4, xpoints4, label="N = 100")
pylab.xlabel("t")
pylab.ylabel("x(t)")
pylab.legend()
pylab.title("Runge-Kutta 4 for -x**3 + sin(t)")
pylab.show()


# In-Class exercise 3: Plotting the output of RK2.f90

rk2_results = "/home/rachel-wood/h_PHYS4840_labs/Labs/rk2_results.dat"
rk2_results_2 = "/home/rachel-wood/h_PHYS4840_labs/Labs/rk2_results_2.dat"

t1, k1 = np.loadtxt(rk2_results, usecols=(0, 1), skiprows=1, unpack=True)
t2, k2 = np.loadtxt(rk2_results_2, usecols=(0, 1), skiprows=1, unpack=True)

# This is my plot for n = int((t_end - t) / dt):
plt.plot(t1, k1)
plt.xlim(0, 10)
plt.title("RK2 for n = int((t_end - t) / dt)")
plt.xlabel("t")
plt.ylabel("k")
#plt.show()

# This is my plot for n = 10000
plt.plot(t2, k2)
plt.xlim(0, 10)
plt.title("RK2 for n = 10000")
plt.xlabel("t")
plt.ylabel("k")
#plt.show()