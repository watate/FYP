#This script is to create the dataframes that will eventually be used by analysis_compare_graph
#basically, what you want to do is:
	#1. Get data from all the models you have (Simple World, Complex World, Simple World with jerk)
	#2. Group data together according to the variable (e.g. Reward, PIDRate, etc.)
		#Each variable should get its own dataframe
	#3. Save dataframes to dataframe folder with appropriate variable names

#If more dataframes are added, locations where we need to add more code is indicated by #ADD_HERE

import pickle
import os
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sb
import pandas as pd

##############################################################################
#Prepare and choose datasets from which to generate graphs
filenames = ["20200302-044536_24000s_dataframe.dat", "20200301-074856_34000c_dataframe.dat"] #ADD_HERE

#Loading the DataFrame generated from preprocess (analysis_tf_summary.py)
#ADD_HERE
if os.path.exists(filenames[0]):
	with open(filenames[0], 'rb') as rfp:
		df0 = pickle.load(rfp)
if os.path.exists(filenames[1]):
	with open(filenames[1], 'rb') as rfp:
		df1 = pickle.load(rfp)

df_list = [df0, df1] #ADD_HERE

#Rename columns
#https://stackoverflow.com/questions/11346283/renaming-columns-in-pandas
for df, i in zip(df_list, range(len(filenames))):
	df = df.rename(columns={'q_max': 'q_max'+ str(i)}, inplace = True)

#Concatenate DataFrames
# Extracting specific columns: 
	#https://stackoverflow.com/questions/34682828/extracting-specific-selected-columns-to-new-dataframe-as-a-copy
#ADD_HERE
df_qmax = df_list[0].filter(["time"], axis=1)
df_qmax = df_qmax.assign(qmax0=df0["q_max0"].copy())
df_qmax = df_qmax.assign(qmax1=df1["q_max1"].copy())

#Rename columns again
df_qmax.columns = ["Time", "Simple World", "Complex World"] #ADD_HERE

#Drop rows that have NaN (i.e. follow the dataframe with the least number of rows)
df_qmax = df_qmax.dropna().copy()

######################## DATA PREPROCESSING (PD.MELT) #########################
with open('dataframes/qmax_simpleVScomplex.dat','wb') as wfp:
	pickle.dump(df_qmax, wfp)