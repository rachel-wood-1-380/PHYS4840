#!/usr/bin/python3


# In-class excercise 1:
import numpy as np
import matplotlib.pyplot as plt
import sys

def dist_mod(d_in_pc):
	modulus = 5 * np.log10(d_in_pc / 10)

	return modulus
# I have put this in my functions library


# In-class exercise 2:

print(dist_mod(8630))
# The distance modulus for NGC 6341 is 14.68.
# We will add this to the isochrome data to line up the plots

load_file = '/home/rachel-wood/h_PHYS4840_labs/Data/MIST_data.dat'

log10_isochrone_age_yr, F606, F814,\
logL, logTeff, phase= np.loadtxt(load_file, usecols=(1,14,18,6,4,22), unpack=True, skiprows=14)

age_Gyr_1e9 = (10.0**log10_isochrone_age_yr)/1e9
age_Gyr = age_Gyr_1e9

age_selection = np.where((age_Gyr > 12) & (age_Gyr <= 13.8)) 

color_selected = F606[age_selection]-F814[age_selection]
magnitude_selected = F606[age_selection]

Teff = 10.0**logTeff
Teff_for_desired_ages =  Teff[age_selection]
logL_for_desired_ages =  logL[age_selection]

phases_for_desired_age = phase[age_selection]
desired_phases = np.where(phases_for_desired_age <= 3)

## now, we can restrict our equal-sized arrays by phase
cleaned_color = color_selected[desired_phases]
cleaned_magnitude = magnitude_selected[desired_phases]
cleaned_Teff = Teff_for_desired_ages[desired_phases]
cleaned_logL = logL_for_desired_ages[desired_phases]

filename = '/home/rachel-wood/h_PHYS4840_labs/Data/NGC6341.dat'

## # Col.  9: F336W calibrated magnitude
## # Col. 15: F438W calibrated magnitude
## # Col. 27: F814W calibrated magnitude
## # Col. 33: membership probability
## but Python indexes from 0, not 1!

blue, green, red, probability = np.loadtxt(filename, usecols=(8, 14, 26, 32), unpack=True)

magnitude = blue
color     = blue - red

quality_cut = np.where( (red   > -99.) &\
					    (blue  > -99)  &\
					    (green > -99)  &\
					    (probability != -1))
 

load_file = '/home/rachel-wood/h_PHYS4840_labs/Data/MIST_data.dat'

log10_isochrone_age_yr, F606, F814,\
logL, logTeff, phase= np.loadtxt(load_file, usecols=(1,14,18,6,4,22), unpack=True, skiprows=14)

age_Gyr_1e9 = (10.0**log10_isochrone_age_yr)/1e9
age_Gyr = age_Gyr_1e9

age_selection = np.where((age_Gyr > 12) & (age_Gyr <= 13.8)) 

color_selected = F606[age_selection]-F814[age_selection]
magnitude_selected = F606[age_selection]

Teff = 10.0**logTeff
Teff_for_desired_ages =  Teff[age_selection]
logL_for_desired_ages =  logL[age_selection]

phases_for_desired_age = phase[age_selection]
desired_phases = np.where(phases_for_desired_age <= 3)

## now, we can restrict our equal-sized arrays by phase
cleaned_color = color_selected[desired_phases]
cleaned_magnitude = magnitude_selected[desired_phases]
cleaned_Teff = Teff_for_desired_ages[desired_phases]
cleaned_logL = logL_for_desired_ages[desired_phases]

####################################################################

'''
Now, recall how we loaded, cleaned, and plotted NGC6341.dat
'''
filename = '/home/rachel-wood/h_PHYS4840_labs/Data/NGC6341.dat'

blue, green, red, probability = np.loadtxt(filename, usecols=(8, 14, 26, 32), unpack=True)

magnitude = blue
color     = blue - red

quality_cut = np.where( (red   > -99.) &\
					    (blue  > -99)  &\
					    (green > -99)  &\
					    (probability != -1))
 
fig, ax = plt.subplots(figsize=(6, 9))  # Single panel

def format_axes(ax):
    ax.tick_params(axis='both', which='major', labelsize=14, length=6, width=1.5)  # Larger major ticks
    ax.tick_params(axis='both', which='minor', labelsize=12, length=3, width=1)    # Minor ticks
    ax.minorticks_on()  # Enable minor ticks

# Plot Isochrone model
ax.plot(cleaned_color, cleaned_magnitude + 14.68, 'go', markersize=2, linestyle='-', color='darkorchid', label='Isochrone Model')

# Plot HST Data
ax.plot(color[quality_cut], magnitude[quality_cut], "k.", markersize=4, alpha=0.2, label='HST Data (NGC6341)', zorder=0)

# Axis settings
ax.invert_yaxis()
ax.set_xlabel('Color', fontsize=18)
ax.set_ylabel('Magnitude', fontsize=18)
ax.set_title('Comparison of Isochrone Model and HST Data', fontsize=16)
ax.set_xlim(0.2,2.5)
ax.set_ylim(26,12)
format_axes(ax)
#ax.legend(fontsize=14, loc='best')

plt.tight_layout()

plt.plot()
plt.show()
#plt.savefig("overlay.png", dpi=300)
#plt.close()

