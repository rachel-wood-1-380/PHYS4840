import numpy as np
import matplotlib.pyplot as plt
import sys

sys.path.append('../')
import h_my_functions_lib as mfl


MIST_file = '/home/rachel-wood/h_PHYS4840_labs/MIST_data.dat'
NGC6341_file = '/home/rachel-wood/h_PHYS4840_labs/NGC6341.dat'

blue, green, red = np.loadtxt(NGC6341_file, usecols=(8, 14, 26), unpack=True)












# Problem 2:

#x = np.linspace(-100, 100, 5000)
#logx = np.sign(x) * np.log10(np.abs(x) + 1e-10)
#y = x**4
#logy = np.log10(y + 1e-10)



#ax[0].plot(x, y, color='r', linestyle=':', linewidth=3)
#ax[0].set_xlim(-100, 100)
#ax[0].set_xlabel("x (linear)", fontsize=16)
#ax[0].set_ylabel("$y = x**4$ (linear)", fontsize=16)
#ax[0].grid(True)

#ax[1].set_xscale('symlog', linthresh=1)
#ax[1].set_yscale('log')
#ax[1].loglog(x, y, color='b', linestyle='--', linewidth=3)
#ax[1].set_xlim(-100, 100)
#ax[1].set_xlabel("x (log)", fontsize=16)
#ax[1].set_ylabel("$y = x**4$ (log)", fontsize=16)
#ax[1].grid(True)
#ax[1].set_title("$y = x**4$", fontsize=20)

#ax[2].plot(logx, logy, color='g', linestyle='-.', linewidth=3)
#ax[2].set_xlim(-2, 2)
#ax[2].set_xlabel("log10 x", fontsize=16)
#ax[2].set_ylabel("log10 y = x**4", fontsize=16)
#ax[2].grid(True)

#plt.tight_layout()
#plt.show()




########################################## Problem 3:
########### Part a)

sunspots_data = '/home/rachel-wood/h_PHYS4840_labs/sunspots.txt'
month, spots = np.loadtxt(sunspots_data, usecols=(0, 1), unpack=True)

years = 1749 + (month - 1) / 12  

fig, ax = plt.subplots(1, 1, figsize=(10, 5))
ax.plot(years, spots, color='goldenrod')

tick_interval_months = 600  
tick_interval_years = tick_interval_months / 12

x_ticks = np.arange(years[0], years[-1], tick_interval_years)
ax.set_xticks(x_ticks)
ax.set_ylim(0, 260)

ax.set_xlabel("Year")
ax.set_ylabel("Number of Sunspots")
ax.set_title("Sunspot Count Per Month")

ax.grid(True)
plt.show()


########### Part b)

sunspots_data = '/home/rachel-wood/h_PHYS4840_labs/sunspots.txt'
month, spots = np.loadtxt(sunspots_data, usecols=(0, 1), unpack=True)

month = month[:1000]
spots = spots[:1000]

years = 1749 + (month - 1) / 12  

# Create plot
fig, ax = plt.subplots(1, 1, figsize=(10, 5))
ax.plot(years, spots, color='goldenrod')

tick_interval_years = 20
x_ticks = np.arange(years[0], years[-1], tick_interval_years)

ax.set_xticks(x_ticks)
ax.set_xticklabels([f"{int(y)}" for y in x_ticks])

ax.set_xlabel("Year")
ax.set_ylabel("Number of Sunspots")
ax.set_title("Sunspot Count Per Month")

ax.grid=(True)
plt.show()


########### Part b)

sunspots_data = '/home/rachel-wood/h_PHYS4840_labs/sunspots.txt'
month, spots = np.loadtxt(sunspots_data, usecols=(0, 1), unpack=True)

month = month[:1000]
spots = spots[:1000]

years = 1749 + (month - 1) / 12  

def calculate_Yk(data, r):

	n = len(data)
	Yk_values = []
	for k in range(n):
		sum_Yk_m = 0
		for m in range(-r, r + 1):
			i = k+m
			if 0 <= i < n:
				sum_Yk_m += data[i]
		Yk = (1 / (2*r)) * sum_Yk_m
		Yk_values .append(Yk)
	return Yk_values

fig, ax = plt.subplots(1, 1, figsize=(10, 5))
ax.plot(years, spots, color='goldenrod', label="Sunspot Count")

running_average = calculate_Yk(spots, 5)

ax.plot(years, running_average, label="running average")

tick_interval_years = 20
x_ticks = np.arange(years[0], years[-1], tick_interval_years)

ax.set_xticks(x_ticks)
ax.set_xticklabels([f"{int(y)}" for y in x_ticks])

ax.set_xlabel("Year")
ax.set_ylabel("Number of Sunspots")
ax.set_title("Sunspot Count Per Month - Running Average")
ax.legend()

ax.grid=(True)
plt.show()