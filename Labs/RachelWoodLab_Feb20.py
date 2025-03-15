#!/usr/bin/python3
########################################################################


import numpy as np
import matplotlib.pyplot as plt

from my_functions_lib import trapezoidal_rule
from my_functions_lib import simpsons_rule
from my_functions_lib import romberg_rule
from my_functions_lib import gauss_legendre_quadrature

def f(x):
    return x**2  # Function to integrate

# **Gauss-Legendre Quadrature**
def gauss_legendre_quadrature(f, a, b, N):
    root, weights = np.polynomial.legendre.leggauss(N)
    integral_approx = 0
    for i in range(N):
        xi = 0.5 * ((b - a) * root[i] + (b + a))  # Transform to [a,b]
        integral_approx += weights[i] * f(xi)
    integral_approx *= 0.5 * (b - a)  # Scaling factor
    return integral_approx


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

# **Romberg Integration**
def romberg_rule(f, a, b, max_order):
    R = np.zeros((max_order, max_order))  # Romberg Table
    
    for i in range(max_order):
        n_intervals = 2**i  # Ensure N is a power of 2
        R[i, 0] = trapezoidal_rule(f, a, b, n_intervals)  # Compute base Trapezoidal value
        
        for j in range(1, i + 1):
            R[i, j] = R[i, j - 1] + (R[i, j - 1] - R[i - 1, j - 1]) / (4**j - 1)
    
    return R[max_order - 1, max_order - 1]


def timing_function(integration_method, x_values, y_values, integral_arg):
    """
    Times the execution of an integration method.

    Parameters:
        integration_method (function): The numerical integration function.
        x_values (array-like): The x values.
        y_values (array-like): The corresponding y values.
        integral_arg (int, optional): EITHER Number of intervals to use (Simpson/Trapz) OR the maximum order of extrapolation (Romberg).

    Returns:
        tuple: (execution_time, integration_result)
    """
    start_time = time.perf_counter()
    result = integration_method(y_values, x_values, integral_arg)
    end_time = time.perf_counter()
    
    return end_time - start_time, result



# **Testing**
N_values = np.array([5, 10, 100, 1000])
exact_value = 2/3

quad_approx = np.zeros(len(N_values))
quad_error = np.zeros(len(N_values))

trap_approx = np.zeros(len(N_values))
trap_error = np.zeros(len(N_values))

simp_approx = np.zeros(len(N_values))
simp_error = np.zeros(len(N_values))

romb_approx = np.zeros(len(N_values))
romb_error = np.zeros(len(N_values))

print("Gauss-Legendre Quadrature:")
for i, N in enumerate(N_values):
    quad_approx[i] = gauss_legendre_quadrature(f, -1, 1, N)
    quad_error[i] = abs(exact_value - quad_approx[i])
    print(f"N={N}, Approx: {quad_approx[i]}, Error: {quad_error[i]}")

print("\nTrapezoidal Rule:")
for i, N in enumerate(N_values):
    trap_approx[i] = trapezoidal_rule(f, -1, 1, N)
    trap_error[i] = abs(exact_value - trap_approx[i])
    print(f"N={N}, Approx: {trap_approx[i]}, Error: {trap_error[i]}")

print("\nSimpson's Rule:")
for i, N in enumerate(N_values):
    simp_approx[i] = simpsons_rule(f, -1, 1, N)
    simp_error[i] = abs(exact_value - simp_approx[i])
    print(f"N={N}, Approx: {simp_approx[i]}, Error: {simp_error[i]}")

print("\nRomberg Integration:")
for i, N in enumerate(N_values):
    if np.log2(N).is_integer():  # Ensure N is a power of 2
        max_order = int(np.log2(N))
    else:
        max_order = 5  # Default to 5 if N is not a power of 2

    romb_approx[i] = romberg_rule(f, -1, 1, max_order)
    romb_error[i] = abs(exact_value - romb_approx[i])
    print(f"N={N}, Approx: {romb_approx[i]}, Error: {romb_error[i]}")



# **Plotting Errors**
plt.figure(figsize=(8,6))
plt.plot(N_values, quad_error, 'o-', label="Gauss-Legendre")
plt.plot(N_values, trap_error, 's-', label="Trapezoidal")
plt.plot(N_values, simp_error, '^-', label="Simpson's")
plt.plot(N_values, romb_error, 'd-', label="Romberg")

plt.xscale("log")  # Log scale for better visualization
plt.yscale("log")  # Log scale for error decay
plt.xlabel("N (Number of Points)")
plt.ylabel("Error")
plt.legend()
plt.grid(True, which="both", linestyle="--", linewidth=0.5)
plt.title("Integration Errors vs. N")
plt.show()
