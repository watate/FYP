#Script 2: Grab preprocessed velocity list data from dataframe_motion, turn it into useful acceleration and jerk data
	#This is a preprocessing script. It does the following:
	#1. Loops through velocity data
	#2. Creates acceleration list
	#3. Loops through acceleration data
	#4. Creates jerk list
	#5. Saves velocity data, acceleration data, and jerk data to the same dataframe

import dill
import cPickle as pickle
import os
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sb
import pandas as pd
import math

############### VELOCITY PROCESSING ########################
#filename = 'normal_complex_velocity_list.dat'
#filename = 'vsmooth_complex_velocity_list.dat'
#filename = 'jerk_complex_velocity_list.dat'
#filename = 'normal_simple_velocity_list.dat'
#filename = 'vsmooth_simple_velocity_list.dat'
filename = 'jerk_simple_velocity_list.dat'

filename1 = filename

filename2 = 'dataframe_motion/' + 'v_' + filename

if os.path.exists(filename2):
	with open(filename2, 'rb') as rfp:
		df = pickle.load(rfp)

################# ADD ACCELERATIONS ##############################

linear_accel = [df["linear_velocity"][i+1] - df['linear_velocity'][i] 
				if df["episode"][i+1] == df["episode"][i]
				else np.nan
				for i in range(0, len(df["linear_velocity"]) - 1)]
linear_accel.insert(0, np.nan)

df['linear_accel'] = linear_accel

angular_accel = [df["angular_velocity"][i+1] - df['angular_velocity'][i] 
				if df["episode"][i+1] == df["episode"][i]
				else np.nan
				for i in range(0, len(df["angular_velocity"]) - 1)]
angular_accel.insert(0, np.nan)

df['angular_accel'] = angular_accel

################# ADD JERK ##############################

linear_jerk = [df["linear_accel"][i+1] - df['linear_accel'][i] 
				if (df["episode"][i+1] == df["episode"][i] and not (math.isnan(df['linear_accel'][i])))
				else np.nan
				for i in range(0, len(df["linear_accel"]) - 1)]
linear_jerk.insert(0, np.nan)
df['linear_jerk'] = linear_jerk

angular_jerk = [df["angular_accel"][i+1] - df['angular_accel'][i] 
				if (df["episode"][i+1] == df["episode"][i] and not (math.isnan(df['angular_accel'][i])))
				else np.nan
				for i in range(0, len(df["angular_accel"]) - 1)]
angular_jerk.insert(0, np.nan)
df['angular_jerk'] = angular_jerk

with open('dataframe_motion/' + 'complete_' + filename1[0:11] + '.pickle','wb') as wfp:
	dill.dump(df, wfp)