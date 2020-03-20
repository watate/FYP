#Script 5: Helps us decide which episode to take for each model
	#It does the following:
		#1. Take in motion data
		#2. Look at episode length (number of steps)
		#3. Choose trajectory with roughly same number of steps

import cPickle as pickle
import os
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd

#filenames = ["complete_normal_simp.dat", "complete_vsmooth_sim.dat", "complete_jerk_simple.dat"]
filenames = ['complete_normal_comp.dat', 'complete_vsmooth_com.dat', 'complete_jerk_comple.dat']
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

df0 = df0.loc[df0['episode'] == 230]
df1 = df1.loc[df1['episode'] == 210]
df2 = df2.loc[df2['episode'] == 250]

# df0 = pd.melt(df0.loc[df0['episode'] == 230])
# df1 = pd.melt(df1.loc[df1['episode'] == 210])
# df2 = pd.melt(df2.loc[df2['episode'] == 250])

variable = 'linear_velocity'

df_new = pd.DataFrame(df0.filter([variable], axis=1))
df_new = df_new.assign(avp1=df1[variable].copy())
df_new = df_new.assign(avp2=df2[variable].copy())

#Rename columns again
df_new.columns = ["Normal Model", "Model with Velocity Smoother", "Model with Jerk-Learning"] #ADD_HERE


# ############## INITIAL SETTINGS ##################
# #prevent xlabel cutting off
# 	#see: https://stackoverflow.com/questions/6774086/why-is-my-xlabel-cut-off-in-my-matplotlib-plot
# from matplotlib import rcParams
# rcParams.update({'figure.autolayout': True})

# #Setting the style
# sns.set(style="darkgrid")


# ############## PAPER ##################
# #Set Paper as format for graphs
# sns.set_context("paper") #other formats: "talk", "poster"

# #Set save directory and figure dpi
# save_dir = "pictures/paper/"
# fig_dpi = 300


# ############## GRAPH SETTINGS ##################
# world = 'complex'
# name = 'Angular Velocity'

# print(df)

# ############## PLOT HISTOGRAM ##################
# #Plot and save graph
# this = "Time"
# that = "value"
# name = "Success Rate"
# world = "Simple World"
# raw_or_smooth = "Smooth"
# g = sb.lineplot(x=this, y=that, hue='variable', legend='full', data=df)
# plt.ylabel("Value")
# plt.xlabel("Time (episodes)") #for averaged results
# plt.title("{0} vs {1} ({2}) ({3})".format(name, this, world, raw_or_smooth))
# fig = g.get_figure()
# fig.savefig(save_dir + "paper_" + world + "_" + name + "_" + raw_or_smooth + "_line" + ".png", dpi=fig_dpi)
# plt.cla()
# plt.clf()
# plt.close("all")