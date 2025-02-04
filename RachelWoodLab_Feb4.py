#!usr/local/Anaconda2023/bin

###################################
#
# Class 5 in-class lab (and notes)
#
#
#to take log base 10 of something, use: np.log10()
#
#
#

import numpy as np
import matplotlib.pyplot as plt
import my_functions_lib

# define your x values
x = np.linspace(1, 100, 500)  # x values

y = 2*x**3

log_x = np.log10(x)
log_y = np.log10(y)

# (1):

plt.plot(x, y)
plt.show()

# (2):
plt.plot(x, y)
plt.xscale('log')
plt.yscale('log')
plt.show()

# (3):

plt.plot(log_x, log_y)
plt.show()