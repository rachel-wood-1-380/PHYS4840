#!/usr/bin/python

import numpy as np




def dist_mod(d_in_pc):
	modulus = 5 * np.log10(d_in_pc / 10)

	return modulus