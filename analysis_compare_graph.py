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
if os.path.exists("dataframes/qmax_simpleVScomplex.dat"):
    with open(filename, 'rb') as rfp:
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
save_dir = "pictures/compare/"
fig_dpi = 300

##############################################################################
#Apply pandas melt
df = pd.melt(df, id_vars=["Time"])

###################### PLOT Q_MAX ##############################
#Plot and save graph
this = "Time"
that = "value"
g = sb.lineplot(x=this, y=that, hue='variable', 
             data=df)
g.set_title("{0} vs {1}".format(this, that))
fig = g.fig
fig.savefig(save_dir + "paper_" + i + "_line" + ".png", dpi=fig_dpi)
plt.cla()
plt.clf()
plt.close("all")

###################### PLOT AVERAGE PERFORMANCE ##################################
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