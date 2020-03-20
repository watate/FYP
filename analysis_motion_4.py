#Script 4: This script generates a multi-plot grid from 3 different dataframes
	#It does the following:
	#1. Pickle in the dataframe with data from 3 models
	#2. Pandas.melt the datasets
	#3. Generate multi-plot grid

import pickle
import os
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sb
import pandas as pd

filenames = ["complete_normal_simp.dat", "complete_vsmooth_sim.dat", "complete_jerk_simple.dat"]
filenames = ["dataframe_motion/" + i for i in filenames]

if os.path.exists(filenames[0]):
	with open(filenames[0], 'rb') as rfp:
		df0 = pickle.load(rfp)
if os.path.exists(filenames[1]):
	with open(filenames[1], 'rb') as rfp:
		df1 = pickle.load(rfp)
if os.path.exists(filenames[2]):
	with open(filenames[2], 'rb') as rfp:
		df2 = pickle.load(rfp)