#!/usr/bin/python3
#########################################

# Class 13

import numpy as np
from numpy import array,empty

A = np.array([ [2,1,4,1], 
			[3,4,-1,-1], 
			[1,-4,1,5], 
			[2,-2,1,3] ],float)

vector = np.array([-4,3,9,7],float)

## dimension 
N = len(vector)

for m in range(N):

	## first, divide by the diagonal element
	divisor = A[m,m]

	## divide every entry in row m by the divisor
	A[m,:] /= divisor

	## the above is shorthand for this operation:
	## A[m,:] = A[m,:]/divisor

	##anything we do to the matrix we must do to the vector:
	vector[m] /= divisor

	## now subtract multipls of the top row from the lower rows
	## to zero out rows 2,3 and 4
	for i in range(m+1, N): ## note that we start from the second row: m+1

		## because the first row now has 1 in the upper-left corner,
		## the factor by which we have to multiply the first row to subtract
		## it from the second is equal to the value in the first entry
		## of the second row
		multiplication_factor = A[i,m] 

		## now we must apply this operation to the entire row 
		## AND vector, as usual 
		A[i,:]    -= multiplication_factor*A[m,:]
		vector[i] -= multiplication_factor*vector[m] 


print('the upper diagonal version of A is: \n', A)

## Write the next part of this program:
##  how do we solve the system of equations now that we have
##  an upper-diagonal matrix?

## you may consult example 6.1 in your textbook if you need help

u = empty(N, float)
for m in range(N-1, -1, -1):
	u[m] = vector[m]
	for i in range(m+1, N):
		u[m] -= A[m, i]*u[i]

print(u)


###########################################################3
# Part 2:

A = array([[2, 1, 4, 1], 
            [3, 4, -1, -1], 
            [1, -4, 1, 5], 
            [2, -2, 1, 3]], float)

## dimension 
N = len(A)

# Initialize L as the N=4 identity matrix 
L = np.array([[1.0 if i == j else 0.0 for j in range(N)] for i in range(N)])
# this above is just a more explicit way of doing
#L = np.identity(N)

print("L looks like this: ", L) ## should return the N=4 I


# initalize U as a copy of A
U = A.copy()


## this double loop will transform L
## into the lower-diagonal form we need
for m in range(N):
    for i in range(m+1, N):        
        
        # Compute the multiplier for the current row operation
        L[i, m] = U[i, m] / U[m, m]
        
        # Subtract the appropriate multiple of the pivot row from the current row
        U[i, :] -= L[i, m] * U[m, :]

print('The lower triangular matrix L is:\n', L)
print('The upper triangular matrix U is:\n', U)

## Write the next part of this program:
##  How do we solve the system of equations using forward and backward substitution?
##  Use L and U to solve Ax = b for a given vector b.

## HINT: see the end of 6.1.4 in your textbook (equations 6.37 through 6.39 in my version)


