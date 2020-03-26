#Script 4: This script generates a single-plot from a selected dataframe generated in analysis_motion_3
	#It does the following:
	#1. Pickle in the single dataframe with motion data from a model
	#2. Generate histogram

import pickle
import os
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sb
import pandas as pd

############## INITIAL SETTINGS ##################
#prevent xlabel cutting off
	#see: https://stackoverflow.com/questions/6774086/why-is-my-xlabel-cut-off-in-my-matplotlib-plot
from matplotlib import rcParams
rcParams.update({'figure.autolayout': True})

#Initial settings
sb.set(style="darkgrid")

#Loading the DataFrame
filepath = "dataframe_motion/complete_normal_simp.dat"
if os.path.exists(filepath):
    with open(filepath, 'rb') as rfp:
        df = pickle.load(rfp)

save_dir = "pictures/paper/"
fig_dpi = 300

############## PLOT HISTOGRAM ##################
#Plot and save graph
#Generating histograms
g = sb.distplot(df["angular_jerk"], kde = False)
fig = g.get_figure()
plt.ylabel("Frequency")
plt.xlabel("Angular Jerk") #for averaged results
plt.title("Histogram of Angular Jerk (Normal Model, Complex World)")
fig.savefig(save_dir + "angular_jerk_normal_complex_hist" + ".png", dpi=fig_dpi)
plt.cla()
plt.clf()
plt.close("all")
