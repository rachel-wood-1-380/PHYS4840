#!/usr/bin/python3
#########################################


# Linear Algebra fact: The determinant of a product of matrices is equal to the product of the determinants.


# eigenvalues.py:

import numpy as np
from numpy import array,empty

A = np.array([ [2, -1, 3,],\
			   [-1, 4, 5], 
			   [3,  5, 6] ],float)

eigenvector_1 =  np.array([-0.5774,\
						   -0.5774,\
						   0.5774],float)

LHS = np.dot(A, eigenvector_1)

## Bonus: Why doesn't this line work??
#LHS = A*eigenvector_1

RHS = -2.0*eigenvector_1

print("LHS:\n",LHS, "\n\nRHS:\n",RHS)


#QR_decomposition.py:

'''
by importing and using the QR decomposition 
algorithm in my_functions_lib.py:
1) Find Q and R
2) Confirm that Q is orthogonal
3) Confirm that R is upper triangular
4) Confirm that the matrix A introduced in eigenvalues.py
can indeed be reconstructed by the dot product 
of matrices Q and R
'''

def qr_decomposition(A):
    ## Computes the QR decomposition of matrix A using
    ## Gram-Schmidt orthogonalization.
    m, n = A.shape
    Q = np.zeros((m, n))
    R = np.zeros((n, n))

    for j in range(n):
        v = A[:, j]  # Take column j of A
        for i in range(j):  # Subtract projections onto previous Q columns
            R[i, j] = np.dot(Q[:, i], A[:, j])
            v = v - R[i, j] * Q[:, i]
        R[j, j] = np.linalg.norm(v)  # Compute norm
        Q[:, j] = v / R[j, j]  # Normalize

    return Q, R

# 1):

Q, R = qr_decomposition(A)
print(Q, R)

# 2): 

QT = np.transpose(Q)

print(np.dot(QT, Q))

# 3):

print(R)

# 4):

print(np.dot(Q, R))