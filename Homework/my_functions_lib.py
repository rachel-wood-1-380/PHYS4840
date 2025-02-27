#!/usr/bin/python3

import numpy as np
import time



def dist_mod(d_in_pc):
	modulus = 5 * np.log10(d_in_pc / 10)

	return modulus


import math
def function2(x):
	return 1 + 1/2 * math.tan(2*x)



# **Trapezoidal Rule**
def trapezoidal_rule(f, a, b, N):
    h = (b - a) / N
    integral = 0.5 * (f(a) + f(b))
    for k in range(1, N):
        xk = a + k * h
        integral += f(xk)
    return integral * h


def trapezoidal_rule_2(x, y):
    N = len(x) - 1
    h = np.diff(x)
    integral = np.sum((y[:-1] + y[1:]) * h / 2)
    return integral



# **Simpson’s Rule**
def simpsons_rule(f, a, b, N):
    if N % 2 == 1:  # Ensure N is even
        N += 1
    h = (b - a) / N
    integral = f(a) + f(b)
    for k in range(1, N, 2):
        xk = a + k * h
        integral += 4 * f(xk)
    for k in range(2, N-1, 2):
        xk = a + k * h
        integral += 2 * f(xk)
    return (h / 3) * integral


def simpsons_rule_2(x, y):
    N = len(x) - 1
    if N % 2 ==1:
        N -= 1
    h = np.diff(x)[:N:2]
    integral = np.sum(h / 3 * (y[:N:2] + 4 * y[1:N:2] + y[2:N+1:2]))
    return integral



# **Romberg Integration**
def romberg_rule(f, a, b, max_order):
    R = np.zeros((max_order, max_order))
    
    for i in range(max_order):
        n_intervals = 2**i  # Ensure N is even
        R[i, 0] = trapezoidal_rule(f, a, b, n_intervals)
        
        for j in range(1, i + 1):
            R[i, j] = R[i, j - 1] + (R[i, j - 1] - R[i - 1, j - 1]) / (4**j - 1)
    
    return R[max_order - 1, max_order - 1]


def romberg_rule_2(x, y, max_order):
    R = np.zeros((max_order, max_order))

    for i in range(max_order):
        n_intervals = 2**i

        x_interp = np.linspace(x[0], x[-1], n_intervals + 1)
        y_interp = np.interp(x_interp, x, y)
        R[i, 0] = np.trapz(y_interp, x_interp)

        # Richardson extrapolation
        for j in range(1, i + 1):
            R[i, j] = R[i, j - 1] + (R[i, j - 1] - R[i - 1, j - 1]) / (4**j - 1)

    return R[max_order - 1, max_order - 1]



def gauss_legendre_quadrature(f, a, b, N):

    root, weights = np.polynomial.legendre.leggauss(N)

    integral_approx = 0
    for i in range(N):
        xi = 0.5 * ((b - a) * root[i] + (b + a))  # Transform to [a,b]

        integral_approx += weights[i] * f(xi)

    integral_approx *= 0.5 * (b - a)  # Scaling factor

    return integral_approx



def timing_function(integration_method, f, a, b, integral_arg):
    """
    Times the execution of an integration method.

    Parameters:
        integration_method (function): The numerical integration function.
        f (function): The integrand function.
        a (float): The lower limit of integration.
        b (float): The upper limit of integration.
        integral_arg (int): The number of intervals (for Simpson/Trapz) or maximum order (for Romberg).

    Returns:
        tuple: (execution_time, integration_result)
    """
    start_time = time.perf_counter()
    result = integration_method(f, a, b, integral_arg)
    end_time = time.perf_counter()
    
    return end_time - start_time, result


def timing_function_romberg(integration_method, f, a, b, max_order):
    """
    Times the execution of an integration method that takes limits a, b, and max_order.
    
    Parameters:
        integration_method (function): The numerical integration function (Romberg).
        f (function): The function to integrate.
        a (float): The lower limit of integration.
        b (float): The upper limit of integration.
        max_order (int): The maximum order for Romberg integration.

    Returns:
        tuple: (execution_time, integration_result)
    """
    start_time = time.perf_counter()
    result = integration_method(f, a, b, max_order)  # Now we pass the function f, a, b, and max_order
    end_time = time.perf_counter()
    
    return end_time - start_time, result

def timing_function_gauss(integration_method, f, a, b, N):
    """
    Times the execution of an integration method that doesn't require y_values and x_values.
    """
    start_time = time.perf_counter()
    result = integration_method(f, a, b, N)  # Pass the function f
    end_time = time.perf_counter()
    
    return end_time - start_time, result