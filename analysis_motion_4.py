#Script 4: This script generates a multi-plot grid from the dataframes generated from analysis_motion_3
	#It does the following:
	#1. Pickle in the single dataframe with data from all 3 models
	#2. Pandas.melt the datasets
	#3. Generate multi-plot grid

import pickle
#import cPickle as pickle
import os
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd

file = 'histogram_angular_velocity_complex_3models.dat'
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
sns.set(style="darkgrid")


############## PAPER ##################
#Set Paper as format for graphs
sns.set_context("paper") #other formats: "talk", "poster"

#Set save directory and figure dpi
save_dir = "pictures/paper/"
fig_dpi = 300

#Apply pandas melt
df = pd.melt(df)

############## GRAPH SETTINGS ##################
world = 'complex'
name = 'Angular Velocity'

print(df)

############## PLOT HISTOGRAM ##################
g = sns.FacetGrid(df, col = "variable")
g.map(plt.hist, "value")
g.set_axis_labels(name, "Frequency")
g.savefig(save_dir + "paper_" + world + "_" + name + "_hist" + ".png", dpi=fig_dpi)