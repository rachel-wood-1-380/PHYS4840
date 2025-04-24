#!/usr/bin/python3
#########################################

import numpy as np
import matplotlib.pyplot as plt

# April 15 In-Class exercise: laplacian operator


###### a):


# The np.roll() function rolls array elements along the specified axis.
# Elements of the input array are being shifted. If an element is being rolled first to the
	# last position, it is rolled back to the first position.



###### b):

# Moved laplacian_operator function to h_my_functions_lib.py


from h_my_functions_lib import laplacian_operator as lapo

N = 30
dx = 1               # Grid spacing
dy = 1
dz = 1
V = 1.0              # Boundary condition
target = 1e-6       # Convergence criterion

# Initialize the potential arrays
phi = np.zeros((N+1, N+1, N+1), dtype=float)
phinew = np.empty_like(phi)

# Apply boundary condition: top face (z = 0) at V, others at 0
phi[:,:,0] = V

# Iterative solution using Gauss-Seidel-like update
delta = 1.0
iteration = 0
while delta > target:
    iteration += 1
    for i in range(1, N):
        for j in range(1, N):
            for k in range(1, N):
                phinew[i,j,k] = (phi[i+dx,j,k] + phi[i-dx,j,k] +
                                 phi[i,j+dy,k] + phi[i,j-dy,k] +
                                 phi[i,j,k+dz] + phi[i,j,k-dz]) / 6.0

    # Preserve boundary conditions
    phinew[:,:,0] = V
    phinew[:,:,N] = 0
    phinew[:,0,:] = 0
    phinew[:,N,:] = 0
    phinew[0,:,:] = 0
    phinew[N,:,:] = 0

    delta = np.max(np.abs(phi - phinew))
    phi, phinew = phinew, phi

 #   if iteration % 10 == 0:
 #       print(f"Iteration {iteration}, max delta = {delta:.2e}")


laplacian = lapo(phi, dx, dy, dz)

# Visualization: middle slice in z-direction
mid_z = N // 2
plt.figure(figsize=(6,5))
plt.imshow(laplacian[:,:,mid_z], origin='lower', cmap='inferno')
plt.colorbar(label='Potential $phi$')
plt.title(f"Midplane slice at z = {mid_z}")
plt.xlabel('x')
plt.ylabel('y')
plt.tight_layout()
plt.show()







####### Example 9.3 from the textbook:

L = 0.01      # Thickness of steel [m]
D = 4.25e-6   # Thermal diffusivity
N = 100       # Number of divisions in grid
a = L / N     # Grid spacing
h = 1e-4      # Time-step
epsilon = h / 1000

T_low = 0.0     # [C]
T_mid = 20.0    # [C]
T_high = 50.0   # [C]

t1 = 0.01
t2 = 0.1
t3 = 0.4
t4 = 1.0
t5 = 10.0
tend = t5 + epsilon

# Create arrays:
T = np.empty(N+1, float)
T[0] = T_high
T[N] = T_low
T[1:N] = T_mid
Tp = np.empty(N+1, float)
Tp[0] = T_high
Tp[N] = T_low

t = 0.0
c = h * D / (a * a)
while t<tend:

	# Calculating the new values of T:
	for i in range(1, N):
		Tp[i] = T[i] + c * (T[i+1]+T[i-1]-2*T[i])
	T, Tp = Tp, T
	t += h


	# Make the plots for the given times:
	if abs(t-t1) < epsilon:
		plt.plot(T)
	if abs(t-t2) < epsilon:
		plt.plot(T)
	if abs(t-t3) < epsilon:
		plt.plot(T)
	if abs(t-t4) < epsilon:
		plt.plot(T)
	if abs(t-t5) < epsilon:
		plt.plot(T)

plt.xlabel("x")
plt.ylabel("T [K]")
plt.show()
