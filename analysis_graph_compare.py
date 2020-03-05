#This script generates graphs that compares trends from different training sessions
#The graphs are generated in folder \compare

import pickle
import os
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sb
import pandas as pd

##############################################################################
#Prepare and choose datasets from which to generate graphs

filename1 = "20200302-044536_24000s_dataframe.dat" #24000s
filename2 = "20200301-074856_34000c_dataframe.dat" #34000c


#prevent xlabel cutting off
	#see: https://stackoverflow.com/questions/6774086/why-is-my-xlabel-cut-off-in-my-matplotlib-plot
from matplotlib import rcParams
rcParams.update({'figure.autolayout': True})

############## VISUALIZATION ##################
#Initial settings
sb.set(style="darkgrid")

#Loading the DataFrame generated from preprocess (analysis_tf_summary.py)
if os.path.exists(filename1):
    with open(filename, 'rb') as rfp:
        df1 = pickle.load(rfp)
if os.path.exists(filename2):
    with open(filename, 'rb') as rfp:
        df2 = pickle.load(rfp)

##############################################################################
#Plot and produce initial raw data
plots = ["reward", "q_max", "pid_rate", "distance", "result", "steps"]
#Set Paper as format for graphs
sb.set_context("paper") #other formats: "talk", "poster"

save_dir = "pictures/raw/"
fig_dpi = 300

#Generating line graphs
for i in plots:
	g = sb.relplot(x="time", y=i, kind="line", data=df)
	#g.fig.autofmt_xdate()
	fig = g.fig
	fig.savefig(save_dir + "paper_" + i + "_line" + ".png", dpi=fig_dpi)
	plt.cla()
	plt.clf()
	plt.close("all")
##############################################################################
##############################################################################
#Divide raw data into:
	#1. Data best represented on line graphs
	#2. Data best represented on histograms
##############################################################################
#Data pre-processing
line_plots = ["average_reward", "average_performance", "average_distance", "average_steps", "average_pid_rate"]

#Obtain raw values from previous dataframe
reward = df["reward"]
q_max = df["q_max"]
distance = df["distance"]
steps = df["steps"]
pid_rate = df["pid_rate"]

#Separate the groups. The last slice will be fine with less than 100 numbers.
groups = [reward[x:x+100] for x in range(0, len(reward), 100)]
#calculate mean
average_reward = [sum(group)/len(group) for group in groups]

#repeat for other variables
groups = [q_max[x:x+100] for x in range(0, len(q_max), 100)]
average_performance = [sum(group)/len(group) for group in groups]
groups = [distance[x:x+100] for x in range(0, len(distance), 100)]
average_distance = [sum(group)/len(group) for group in groups]
groups = [steps[x:x+100] for x in range(0, len(steps), 100)]
average_steps = [sum(group)/len(group) for group in groups]
groups = [pid_rate[x:x+100] for x in range(0, len(pid_rate), 100)]
average_pid_rate = [sum(group)/len(group) for group in groups]
policy_usage = [1 - i for i in average_pid_rate]

data = {"Time": np.arange(len(average_performance)),
		"Average Reward": average_reward,
		"Average Performance": average_performance,
		"Average Distance": average_distance,
		"Average Steps": average_steps,
		"Average PID Rate": average_pid_rate,
		"Policy Usage": policy_usage}

#Generate second dataframe for averaged results
df2 = pd.DataFrame(data, columns=["Time", "Average Reward", "Average Performance", "Average Distance", "Average Steps", "Average PID Rate", "Policy Usage"])
##############################################################################
##############################################################################
#1. Data best represented on line graphs
sb.set_context("paper") #other formats: "talk", "poster"

save_dir = "pictures/paper/"
fig_dpi = 300

line_plots = ["Time", "Average Reward", "Average Performance", "Average Distance", "Average Steps", "Average PID Rate", "Policy Usage"]
#Generating line graphs
for i in line_plots:
	g = sb.relplot(x="Time", y=i, kind="line", data=df2)
	#g.fig.autofmt_xdate()
	g.set(xlabel='Time (x10^2 episodes)')
	fig = g.fig
	fig.savefig(save_dir + "paper_" + i + "_line" + ".png", dpi=fig_dpi)
	plt.cla()
	plt.clf()
	plt.close("all")
##############################################################################
#2. Data best represented on histograms
hist_plots = ["pid_rate", "result"]

#Generating histograms
for i in hist_plots:
	g = sb.distplot(df[i], kde = False)
	fig = g.get_figure()
	fig.savefig(save_dir + "paper_" + i + "_hist" + ".png", dpi=fig_dpi)
	plt.cla()
	plt.clf()
	plt.close("all")

##############################################################################
##############################################################################
#1. Data best represented on line graphs
sb.set_context("talk") #other formats: "talk", "poster"

save_dir = "pictures/talk/"
fig_dpi = 300

#Generating line graphs
for i in line_plots:
	g = sb.relplot(x="Time", y=i, kind="line", data=df2)
	#g.fig.autofmt_xdate()
	g.set(xlabel='Time (x10^2 episodes)')
	fig = g.fig
	fig.savefig(save_dir + "paper_" + i + "_line" + ".png", dpi=fig_dpi)
	plt.cla()
	plt.clf()
	plt.close("all")
##############################################################################
#2. Data best represented on histograms
hist_plots = ["pid_rate", "result"]

#Generating histograms
for i in hist_plots:
	g = sb.distplot(df[i], kde = False)
	fig = g.get_figure()
	fig.savefig(save_dir + "paper_" + i + "_hist" + ".png", dpi=fig_dpi)
	plt.cla()
	plt.clf()
	plt.close("all")

##############################################################################
##############################################################################