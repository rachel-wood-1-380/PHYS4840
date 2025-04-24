#!/usr/bin/python3

#############################################################################################################
#############################################################################################################
#############################################################################################################

############ Imports ############

import numpy as np
from math import tanh, cosh
import fourier_series as fs
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

#############################################################################################################
#############################################################################################################
#############################################################################################################

############ Useful Astronomy Functions ############

def dist_mod(d_in_pc):
	modulus = 5 * np.log10(d_in_pc / 10)

	return modulus

#############################################################################################################
#############################################################################################################
#############################################################################################################

############ Solving ODEs ############

def euler_method(f, x0, t0, t_end, dt): # Define inputs to the function.
	t_values = np.arange(t0, t_end + dt, dt) # A np array with evenly spaced values between t0 and
	# t_end + dt in step sizes dt.
	x_values = np.zeros(len(t_values)) # A numpy array, same size as t_values.
	x_values[0] = x0 # The first x value in the index is equal to the inital x condition.

	for i in range(1, len(t_values)): # Looping through the length of the t values.
		x_values[i] = x_values[i - 1] + dt * f(x_values[i - 1], t_values[i - 1])
        # This calculates the x value at each index. 
        # "f(x_values[i - 1], t_values[i - 1])" is f(x).

	return t_values, x_values

def runge_kutta_2_method(f, x0, t0, t_end, N):
	dt = (t_end - 0) / N
	t_values = np.arange(t0, t_end + dt, dt)
	x_values = np.zeros(len(t_values))
	x_values[0] = x0

	for i in range(1, len(t_values)):
		k1 = dt * f(x_values[i - 1], t_values[i - 1])
		k2 = dt * f(x_values[i - 1] + 0.5 * k1, t_values[i - 1] + 0.5 * dt)
		x_values[i] = x_values[i - 1] + k2

	return t_values, x_values


def runge_kutta_4_method(f, x0, t0, t_end, N):
	dt = (t_end - 0) / N
	t_values = np.arange(t0, t_end + dt, dt)
	x_values = np.zeros(len(t_values))
	x_values[0] = x0

	for i in range(1, len(t_values)):
		k1 = dt * f(x_values[i - 1], t_values[i - 1])
		k2 = dt * f(x_values[i - 1] + 0.5, t_values[i - 1] + 0.5 * dt)
		k3 = dt * f(x_values[i - 1] + 0.5 * k1, t_values[i - 1] + 0.5 * dt)
		k4 = dt * f(x_values[i - 1] + k3, t_values[i - 1] + dt)
		x_values[i] = x_values[i - 1] + (k1 + 2 * k2 + 2 * k3 + k4) / 6

	return t_values, x_values


#############################################################################################################
#############################################################################################################
#############################################################################################################

############ Wave shapes ############

def square_wave(x):
    """Square wave: 1 for 0 <= x < pi, -1 for pi <= x < 2pi"""
    return np.where((x % (2*np.pi)) < np.pi, 1.0, -1.0)


def sawtooth_wave(x):
    """Sawtooth wave: from -1 to 1 over 2pi period"""
    return (x % (2*np.pi)) / np.pi - 1


def triangle_wave(x):
    """Triangle wave with period 2pi"""
    # Normalize to [0, 2pi]
    x_norm = x % (2*np.pi)
    # For 0 to pi, goes from 0 to 1
    # For pi to 2pi, goes from 1 to 0
    return np.where(x_norm < np.pi, 
                   x_norm / np.pi, 
                   2 - x_norm / np.pi)


def pulse_train(x):
    """Pulse train: 1 for small interval, 0 elsewhere"""
    x_norm = x % (2*np.pi)
    pulse_width = np.pi / 8  # Very narrow pulse
    return np.where(x_norm < pulse_width, 1.0, 0.0)


def half_rectified_sine(x):
    """Half-rectified sine wave: max(0, sin(x))"""
    return np.maximum(0, np.sin(x))


def ecg_like_signal(x):

        def r(val, variation=0.1):
            return val * np.random.uniform(1 - variation, 1 + variation)

        # Normalize x to [0, 2pi]
        x_norm = x % (2 * np.pi)
        # P-wave with randomized amplitude, center, and width
        p_wave = r(0.25, 0.2) * np.exp(-((x_norm - r(0.7 * np.pi, 0.05))**2) / (r(0.1 * np.pi, 0.05)**2))
        # QRS complex: one positive peak and two negative deflections
        qrs1 = r(1.0, 0.2) * np.exp(-((x_norm - r(np.pi, 0.05))**2) / (r(0.05 * np.pi, 0.05)**2))
        qrs2 = r(-0.3, 0.2) * np.exp(-((x_norm - r(0.9 * np.pi, 0.05))**2) / (r(0.04 * np.pi, 0.05)**2))
        qrs3 = r(-0.2, 0.2) * np.exp(-((x_norm - r(1.1 * np.pi, 0.05))**2) / (r(0.04 * np.pi, 0.05)**2))
        # T-wave with random parameters
        t_wave = r(0.5, 0.2) * np.exp(-((x_norm - r(1.4 * np.pi, 0.05))**2) / (r(0.1 * np.pi, 0.05)**2))
    
        return p_wave + qrs1 + qrs2 + qrs3 + t_wave


def abssine_wave(x):
	return (abs(np.sin(x)))


#############################################################################################################
#############################################################################################################
#############################################################################################################

############ Numerical Integration Methods ############


def trapezoidal_rule(f, a, b, N):
    """
    Approximates the integral using the trapezoidal rule with a loop.

    Parameters:
        f (function or array-like): A function, it's evaluated at N+1 points.
                                    
        a (float): Lower bound of integration.
        b (float): Upper bound of integration.
        N (int): Number of intervals (trapezoids).

    Returns:
        float: The approximated integral.
    """
    
    h = (b - a) / N

    integral = (1/2) * (f(a) + f(b)) * h  # Matches the first & last term in the sum

    # Loop through k=1 to N-1 to sum the middle terms
    for k in range(1, N):
        xk = a + k * h  # Compute x_k explicitly (matches the formula)
        integral += f(xk) * h  # Normal weight (multiplied by h directly)

    return integral



def simpsons_rule(f, a, b, N):
    """
    Approximates the integral using Simpson's rule.

    Parameters:
        f (function): The function to integrate.
        a (float): Lower bound of integration.
        b (float): Upper bound of integration.
        N (int): Number of intervals (must be even).

    Returns:
        float: The approximated integral.
    """
    if N % 2 != 0:
        raise ValueError("N must be even for Simpson's rule.")

    h = (b - a) / N  # Step size
    integral = f(a) + f(b)  # First and last terms

    # Odd indices (weight 4)
    for k in range(1, N, 2):
        xk = a + k * h
        integral += 4 * f(xk)

    # Even indices (weight 2)
    for k in range(2, N, 2):
        xk = a + k * h
        integral += 2 * f(xk)

    return (h / 3) * integral


def romberg_rule(f, a, b, max_order):
    """
    Approximates the integral using Romberg's method, leveraging the trapezoidal rule.

    Parameters:
        f (function): The function to integrate.
        a (float): Lower bound of integration.
        b (float): Upper bound of integration.
        max_order (int): Maximum order (controls accuracy).

    Returns:
        float: The approximated integral.
    """
    R = np.zeros((max_order, max_order))  # Create a Romberg table
    
    # First approximation using the trapezoidal rule
    R[0, 0] = trapezoidal_rule(f, a, b, 1)
    
    for i in range(1, max_order):
        N = 2**i  # Number of intervals (doubles each step)
        R[i, 0] = trapezoidal_rule(f, a, b, N)
        
        # Compute extrapolated Romberg values
        for j in range(1, i + 1):
            R[i, j] = R[i, j - 1] + (R[i, j - 1] - R[i - 1, j - 1]) / (4**j - 1)
    
    return R[max_order - 1, max_order - 1]  # Return the most refined estimate






def laplacian_operator(Phi, dx, dy, dz):
    """
    Compute the Laplacian of a scalar field Phi (i.e., apply the Poisson operator)
    using central finite differences on a 3D uniform grid.

    Parameters:
    - Phi : 3D numpy array of shape (nx, ny, nz)
    - dx, dy, dz : grid spacings in x, y, z directions

    Returns:
    - laplacian : 3D numpy array of the same shape as Phi
    """

    laplacian = (
        (np.roll(Phi, -1, axis=0) - 2*Phi + np.roll(Phi, 1, axis=0)) / dx**2 +
        (np.roll(Phi, -1, axis=1) - 2*Phi + np.roll(Phi, 1, axis=1)) / dy**2 +
        (np.roll(Phi, -1, axis=2) - 2*Phi + np.roll(Phi, 1, axis=2)) / dz**2
    )

    return laplacian
