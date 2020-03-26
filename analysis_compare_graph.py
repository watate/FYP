#Script 3: This script is used after analysis_compare_tfsummary.
	#We generate graphs that compare the results of different trainings
#This script generates graphs that compares trends from different training sessions
#The graphs are generated in folder \compare

import pickle
import os
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sb
import pandas as pd

##############################################################################
#Prepare and choose dataframe from which to generate graphs
filepath = "dataframes/simple_reward_smooth_3models.dat"
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

#Set Paper as format for graphs
sb.set_context("paper") #other formats: "talk", "poster"

#Set save directory and figure dpi
save_dir = "pictures/updates/"
fig_dpi = 300

##############################################################################
#Apply pandas melt
df = pd.melt(df, id_vars=["Time"])

###################### PLOT Q_MAX ##############################
#Plot and save graph
this = "Time"
xname = "Training Episodes"
that = "value"
name = "Performance"
world = "Simple World"
raw_or_smooth = "Smooth"
g = sb.lineplot(x=this, y=that, hue='variable', legend='full', data=df)
plt.ylabel("Value")
plt.xlabel(xname) #for averaged results
plt.title("{0} vs {1} ({2}) ({3})".format(name, xname, world, raw_or_smooth))
fig = g.get_figure()
fig.savefig(save_dir + "paper_" + world + "_" + name + "_" + raw_or_smooth + "_line" + ".png", dpi=fig_dpi)
plt.cla()
plt.clf()
plt.close("all")

######################## PLOT SUCCESS RATE #####################################

