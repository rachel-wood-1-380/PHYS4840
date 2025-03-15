#!/usr/bin/python3

# Solve: x = 2 - e^{-x}

from math import exp
from math import sqrt
from math import log
import numpy as np

x = 0.5

for i in range(10):
	x = 2 - exp(-x)
#	print(x)

# Form (A):
x = 0.5

for i in range(50):
	x = exp(1.0 - x**2)
	print(x)

# Form (B)

x = 0.5

for i in range(50):
	x = np.sqrt(1.0 - np.log(x))
	print(x)