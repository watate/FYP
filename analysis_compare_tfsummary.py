#This script is to create the dataframes that will eventually be used by analysis_compare_graph
#basically, what you want to do is:
	#1. Get data from all the models you have (Simple World, Complex World, Simple World with jerk)
	#2. Group data together according to the variable (e.g. Reward, PIDRate, etc.)
		#Each variable should get its own dataframe
	#3. Save dataframes to dataframe folder with appropriate variable names
#Script 2: This script is used after analysis_compare_tfsummary_EMA or analysis_tfsummary
	#That is, we only use this script after we have the dataframes and we want to put them together in a combined dataframe

import pickle
import os
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sb
import pandas as pd


#Prepare and choose datasets from which to generate graphs #ADD_HERE
########################### SIMPLE #########################################
#Simple Raw
#filenames = ["20200302-044536_24000s_dataframe.dat", "20200311-004451_VSmooth_Simple_dataframe.dat", "20200309-000445_Jerk_Simple_dataframe.dat"]

#Simple Smooth
#filenames = ["Normal_Simple_smooth.dat", "VSmooth_Simple_smooth.dat", "Jerk_Simple_smooth.dat"]
#filenames = ["dataframe_smoothed/" + i for i in filenames]

########################### COMPLEX #########################################
#Complex Raw
#filenames = ["20200301-074856_34000c_dataframe.dat", "20200311-203436_VSmooth_Complex_dataframe.dat", "20200309-214742_Jerk_Complex_dataframe.dat"]

#Complex Smooth
filenames = ["Normal_Complex_smooth.dat", "VSmooth_Complex_smooth.dat", "Jerk_Complex_smooth.dat"]
filenames = ["dataframe_smoothed/" + i for i in filenames]

#Loading the DataFrame generated from preprocess (analysis_tf_summary.py)
#ADD_HERE
if os.path.exists(filenames[0]):
	with open(filenames[0], 'rb') as rfp:
		df0 = pickle.load(rfp)
if os.path.exists(filenames[1]):
	with open(filenames[1], 'rb') as rfp:
		df1 = pickle.load(rfp)
if os.path.exists(filenames[2]):
	with open(filenames[2], 'rb') as rfp:
		df2 = pickle.load(rfp)

df_list = [df0, df1, df2] #ADD_HERE

#Rename columns
#https://stackoverflow.com/questions/11346283/renaming-columns-in-pandas
#for df, i in zip(df_list, range(len(filenames))):
#	df = df.rename(columns={'q_max': 'q_max'+ str(i)}, inplace = True)

#Concatenate DataFrames
# Extracting specific columns: 
	#https://stackoverflow.com/questions/34682828/extracting-specific-selected-columns-to-new-dataframe-as-a-copy
#ADD_HERE

df_new = pd.DataFrame(df0.filter(["Time"], axis=1))
df_new = df_new.assign(avp0=df0["Success Rate"].copy())
df_new = df_new.assign(avp1=df1["Success Rate"].copy())
df_new = df_new.assign(avp2=df2["Success Rate"].copy())

#Rename columns again
df_new.columns = ["Time", "Normal Model", "Model with Velocity Smoother", "Model with Jerk-Learning"] #ADD_HERE

#Drop rows that have NaN (i.e. follow the dataframe with the least number of rows)
df_new = df_new.dropna().copy()

######################## DATA PREPROCESSING (PD.MELT) #########################
with open('dataframes/complex_success_rate_smooth_3models.dat','wb') as wfp:
	pickle.dump(df_new, wfp)