#Script 4: This script generates a multi-plot grid from the dataframes generated from analysis_motion_3
	#It does the following:
	#1. Pickle in the single dataframe with data from all 3 models
	#2. Pandas.melt the datasets
	#3. Generate multi-plot grid

import pickle
import os
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sb
import pandas as pd

file = 'histogram_linear_jerk_simple_3models.dat'
filepath = 'dataframes/' + file

if os.path.exists(filepath):
    with open(filepath, 'rb') as rfp:
        df = pickle.load(rfp)

############## INITIAL SETTINGS ##################
#prevent xlabel cutting off
	#see: https://stackoverflow.com/questions/6774086/why-is-my-xlabel-cut-off-in-my-matplotlib-plot
from matplotlib import rcParams
rcParams.update({'figure.autolayout': True})

#Setting the style
sb.set(style="darkgrid")


############## PAPER ##################
#Set Paper as format for graphs
sb.set_context("paper") #other formats: "talk", "poster"

#Set save directory and figure dpi
save_dir = "pictures/paper/"
fig_dpi = 300

############## TIDY DATA ##################
#Apply pandas melt
df = pd.melt(df, id_vars=["Time"])

############## PLOT HISTOGRAM ##################
print(df)