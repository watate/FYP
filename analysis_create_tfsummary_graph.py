#Generate graphs from dataframe created in analysis_tf_summary.py
#A lot of code here has been commented out
	#uncomment the code if you would like to make different graphs (e.g. histograms)

#Seaborn Documentation for relplot:
	#https://seaborn.pydata.org/generated/seaborn.relplot.html#seaborn.relplot

import pickle
import os
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sb
import pandas as pd

#filename = 
#filename = "20200226-093050_3000b_dataframe.dat" #Blank World
#filename = "20200226-170933_3000b16000s_dataframe.dat" #Simple World (after training in b)
#filename = "20200227-220051_3000b16000s30000c_dataframe.dat" #Complex World (after training in b and s)
#filename = "20200302-044536_24000s_dataframe.dat" #24000s
filename = "20200301-074856_34000c_dataframe.dat" #34000c

#prevent xlabel cutting off
	#see: https://stackoverflow.com/questions/6774086/why-is-my-xlabel-cut-off-in-my-matplotlib-plot
from matplotlib import rcParams
rcParams.update({'figure.autolayout': True})

############## VISUALIZATION ##################
#Initial settings
sb.set(style="darkgrid")

#Loading the DataFrame generated from preprocess (analysis_tf_summary.py)
if os.path.exists(filename):
    with open(filename, 'rb') as rfp:
        df = pickle.load(rfp)

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
# #name plots
# #these variables are best represented in terms of a line graph
# #plots = ["reward", "q_max", "pid_rate", "distance", "result", "steps"]

# ######### ANGULAR ############
# #Set Paper as format for graphs
# sb.set_context("paper") #other formats: "talk", "poster"

# save_dir = "pictures/paper/"
# fig_dpi = 300

# # #Generating histograms
# # for i in plots:
# # 	g = sb.distplot(df[i], kde = False)
# # 	fig = g.get_figure()
# # 	fig.savefig(save_dir + "paper_" + i + "_hist" + ".png", dpi=fig_dpi)
# # 	plt.cla()
# # 	plt.clf()
# # 	plt.close("all")

# #Generating line graphs
# for i in line_dataframe_names:
# 	g = sb.relplot(x="time", y=i, kind="line", data=df)
# 	#g.fig.autofmt_xdate()
# 	fig = g.fig
# 	fig.savefig(save_dir + "paper_" + i + "_line" + ".png", dpi=fig_dpi)
# 	plt.cla()
# 	plt.clf()
# 	plt.close("all")

# #Generate average performance graph
# g = sb.relplot(x="time", y="average_performance", kind="line", data=df2)
# #g.fig.autofmt_xdate()
# g.set(xlabel='time (x10^2 episodes)', ylabel='average_performance')
# fig = g.fig
# fig.savefig(save_dir + "paper_" + "average_performance" + "_line" + ".png", dpi=fig_dpi)
# plt.cla()
# plt.clf()
# plt.close("all")

# #Set Talk as format for graphs
# sb.set_context("talk") #other formats: "talk", "poster"
# save_dir = "pictures/talk/"

# #Generating line graphs
# for i in plots:
# 	g = sb.relplot(x="time", y=i, kind="line", data=df)
# 	#g.fig.autofmt_xdate()
# 	fig = g.fig
# 	fig.savefig(save_dir + "paper_" + i + "_line" + ".png", dpi=fig_dpi)
# 	plt.cla()
# 	plt.clf()
# 	plt.close("all")

# #Generate average performance graph
# g = sb.relplot(x="time", y="average_performance", kind="line", data=df2)
# #g.fig.autofmt_xdate()
# g.set(xlabel='time (x10^2 episodes)', ylabel='average_performance')
# fig = g.fig
# fig.savefig(save_dir + "paper_" + "average_performance" + "_line" + ".png", dpi=fig_dpi)
# plt.cla()
# plt.clf()
# plt.close("all")

# #Generating histograms
# for i in plots:
# 	g = sb.distplot(df[i], kde = False)
# 	fig = g.get_figure()
# 	fig.savefig(save_dir + "talk_" + i + "_hist" + ".png", dpi=fig_dpi)
# 	plt.cla()
# 	plt.clf()
# 	plt.close("all")

# #Generating line graphs
# for i in plots:
# 	g = sb.relplot(x="time", y=i, kind="line", data=df)
# 	g.fig.autofmt_xdate()
# 	fig = g.fig
# 	fig.savefig(save_dir + "talk_" + i + "_line" + ".png", dpi=fig_dpi)
# 	plt.cla()
# 	plt.clf()
# 	plt.close("all")




# g = sb.distplot(df["angular_velocity"], kde = False)
# plt.show()

# g = sb.distplot(df["angular_jerk"], kde = False)
# plt.show()

# #Generating the graphs
# g = sb.relplot(x="time", y="angular_velocity", kind="line", data=df)
# g.fig.autofmt_xdate()
# plt.show()

# g = sb.relplot(x="time", y="angular_acceleration", kind="line", data=df)
# g.fig.autofmt_xdate()
# plt.show()

# g = sb.relplot(x="time", y="angular_jerk", kind="line", data=df)
# g.fig.autofmt_xdate()
# plt.show()

# sb.set_context("talk")

# g = sb.distplot(df["angular_velocity"], kde = False)
# plt.show()

# g = sb.distplot(df["angular_jerk"], kde = False)
# plt.show()

# #Generating the graphs
# g = sb.relplot(x="time", y="angular_velocity", kind="line", data=df)
# g.fig.autofmt_xdate()
# plt.show()

# g = sb.relplot(x="time", y="angular_acceleration", kind="line", data=df)
# g.fig.autofmt_xdate()
# plt.show()

# g = sb.relplot(x="time", y="angular_jerk", kind="line", data=df)
# g.fig.autofmt_xdate()
# plt.show()

# ########### LINEAR #############
# #Generating histograms
# g = sb.distplot(df["linear_velocity"], kde = False)
# plt.show()

# g = sb.distplot(df["linear_jerk"], kde = False)
# plt.show()

# #Generating the graphs
# g = sb.relplot(x="time", y="linear_velocity", kind="line", data=df)
# g.fig.autofmt_xdate()
# plt.show()

# g = sb.relplot(x="time", y="linear_acceleration", kind="line", data=df)
# g.fig.autofmt_xdate()
# plt.show()

# g = sb.relplot(x="time", y="linear_jerk", kind="line", data=df)
# g.fig.autofmt_xdate()
# plt.show()


# #plot linear velocity
# x = np.linspace(1, len(linear_velocity), len(linear_velocity))
# plt.plot(x, linear_velocity, label="Linear Velocity")
# plt.xlabel('Timestep')
# plt.ylabel('Velocity')
# plt.title('Velocity vs Timestep')
# plt.legend()
# plt.grid()
# plt.show()

# #plot linear acceleration
# x = np.linspace(1, len(linear_acceleration), len(linear_acceleration))
# plt.plot(x, linear_acceleration, label="Linear Acceleration")
# plt.xlabel('Timestep')
# plt.ylabel('Acceleration')
# plt.title('Acceleration vs Timestep')
# plt.legend()
# plt.grid()
# plt.show()

# #plot linear jerk
# x = np.linspace(1, len(linear_jerk), len(linear_jerk))
# plt.plot(x, linear_jerk, label="Linear Jerk")
# plt.xlabel('Timestep')
# plt.ylabel('Jerk')
# plt.title('Jerk vs Timestep')
# plt.legend()
# plt.grid()
# plt.show()

# #plot linear jerk
# x = np.linspace(1, len(linear_jerk), len(linear_jerk))
# plt.plot(x, linear_velocity, label="Linear Velocity")
# plt.plot(x, linear_acceleration, label="Linear Acceleration")
# plt.plot(x, linear_jerk, label="Linear Jerk")
# plt.xlabel('Timestep')
# plt.ylabel('Variable')
# plt.title('Variable vs Timestep')
# plt.legend()
# plt.show()


# #plot linear velocity and angular velocity
# x = np.linspace(1, len(linear_velocity), len(linear_velocity))
# plt.plot(x, linear_velocity, label="Linear Velocity")
# plt.plot(x, angular_velocity, label="Angular Velocity")
# plt.xlabel('Timestep')
# plt.ylabel('Velocity')
# plt.title('Velocity vs Timestep')
# plt.legend()
# plt.show()

# x = np.linspace(1, len(angular_velocity), len(angular_velocity))
# plt.plot(x, angular_velocity)
# plt.xlabel('Timestep')
# plt.ylabel('Angular Velocity')
# plt.title('Angular Velocity vs Timestep')
# plt.show()
