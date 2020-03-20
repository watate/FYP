#Script 1: Grab velocity list data, process it into jerk
	#This is a preprocessing script. It does the following:
	#1. Reorganizes the data into columns
	#2. Labels the columns
	#3. Saves it as a new dataframe in dataframe_motion

import pickle
import os
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sb
import pandas as pd

############### VELOCITY PROCESSING ########################
#filename = 'normal_complex_velocity_list.dat'
#filename = 'vsmooth_complex_velocity_list.dat'
#filename = 'jerk_complex_velocity_list.dat'
#filename = 'normal_simple_velocity_list.dat'
#filename = 'vsmooth_simple_velocity_list.dat'
filename = 'jerk_simple_velocity_list.dat'

if os.path.exists(filename):
	with open(filename, 'rb') as rfp:
		df = pickle.load(rfp)

e_list = list()
l_vel = list()
a_vel = list()

############# FIRST, LABEL DATA #######################
for i in df:
	episode = i[0]
	linear = i[1]
	angular = i[2]
	e_list.append(episode)
	l_vel.append(linear)
	a_vel.append(angular)

data = {"episode" : e_list,
		"linear_velocity" : l_vel,
		"angular_velocity" : a_vel}

df_new = pd.DataFrame(data)

with open('dataframe_motion/' + 'v_' + filename,'wb') as wfp:
	pickle.dump(df_new, wfp)