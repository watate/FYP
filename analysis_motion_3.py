#Script 3: This script combines a desired variable from all 3 models into a single dataframe
 #It does the following:
 	#1. Pickle in all 3 dataframes (3 Models)
 	#2. Select column for desired variable from each of the 3 dataframes
 	#3. Assign the 3 columns to a new dataframe
 	#4. Export the new dataframe to /dataframes

import dill
import pickle
import os
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sb
import pandas as pd

#filenames = ["complete_normal_simp.dat", "complete_vsmooth_sim.dat", "complete_jerk_simple.dat"]
filenames = ['complete_normal_comp.dat', 'complete_vsmooth_com.dat', 'complete_jerk_comple.dat']
filenames = ["dataframe_motion/" + i for i in filenames]
variable = "angular_jerk"
world = 'complex'

if os.path.exists(filenames[0]):
	with open(filenames[0], 'rb') as rfp:
		df0 = pickle.load(rfp)
if os.path.exists(filenames[1]):
	with open(filenames[1], 'rb') as rfp:
		df1 = pickle.load(rfp)
if os.path.exists(filenames[2]):
	with open(filenames[2], 'rb') as rfp:
		df2 = pickle.load(rfp)


#df_new = pd.DataFrame(df0.filter(["episode"], axis=1))
#df_new = df_new.assign(avp0=df0["linear_jerk"].copy())
df_new = pd.DataFrame(df0.filter([variable], axis=1))
df_new = df_new.assign(avp1=df1[variable].copy())
df_new = df_new.assign(avp2=df2[variable].copy())

#Rename columns again
df_new.columns = ["Normal Model", "Model with Velocity Smoother", "Model with Jerk-Learning"] #ADD_HERE

#Drop rows that have NaN (i.e. follow the dataframe with the least number of rows)
df_new = df_new.dropna().copy()

######################## DATA PREPROCESSING (PD.MELT) #########################
with open('dataframes/' 'histogram_' + variable + '_' + world + '_3models.dat','wb') as wfp:
	pickle.dump(df_new, wfp)