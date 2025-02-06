#!usr/local/Anaconda2023/bin/python

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

filename = 'NGC6341.py'

blue, green, red = np.loadtxt("NGC6341.py", usecols=(8, 14, 26), unpack=True)

plt.scatter(blue - red, blue, c="black")
plt.xlabel("Color: B-R")
plt.ylabel("Magnitude: B")
plt.title("Hubble Space Telescope Data for the Globular Cluster NGC6341")

plt.show()

