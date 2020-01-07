import pickle
import os
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sb
import pandas as pd

#prevent xlabel cutting off
	#see: https://stackoverflow.com/questions/6774086/why-is-my-xlabel-cut-off-in-my-matplotlib-plot
from matplotlib import rcParams
rcParams.update({'figure.autolayout': True})

############## VISUALIZATION ##################
#Initial settings
sb.set(style="darkgrid")

#Loading the DataFrame generated from preprocess (analysis2.py)
if os.path.exists('analysis_dataframe.dat'):
    with open('analysis_dataframe.dat', 'rb') as rfp:
        df = pickle.load(rfp)

# #Generating the DataFrame
# df = pd.DataFrame(dict(time=np.arange(len(linear_velocity)),
#                        linear_velocity=linear_velocity,
# 						linear_acceleration=linear_acceleration,
# 						linear_jerk=linear_jerk,
# 						angular_velocity=angular_velocity,
# 						angular_acceleration=angular_acceleration,
# 						angular_jerk=angular_jerk,
# 						))

plots = ["linear_velocity", "linear_acceleration", "linear_jerk", "angular_velocity", "angular_acceleration", "angular_jerk"]

######### ANGULAR ############
#Set Paper
sb.set_context("paper") #other formats: "talk", "poster"

save_dir = "pictures/paper/"
fig_dpi = 300

#Generating histograms
for i in plots:
	g = sb.distplot(df[i], kde = False)
	fig = g.get_figure()
	fig.savefig(save_dir + "paper_" + i + "_hist" + ".png", dpi=fig_dpi)
	plt.cla()
	plt.clf()
	plt.close("all")

#Generating line graphs
for i in plots:
	g = sb.relplot(x="time", y=i, kind="line", data=df)
	#g.fig.autofmt_xdate()
	fig = g.fig
	fig.savefig(save_dir + "paper_" + i + "_line" + ".png", dpi=fig_dpi)
	plt.cla()
	plt.clf()
	plt.close("all")

#Set Talk
sb.set_context("talk") #other formats: "talk", "poster"
save_dir = "pictures/talk/"

#Generating histograms
for i in plots:
	g = sb.distplot(df[i], kde = False)
	fig = g.get_figure()
	fig.savefig(save_dir + "talk_" + i + "_hist" + ".png", dpi=fig_dpi)
	plt.cla()
	plt.clf()
	plt.close("all")

#Generating line graphs
for i in plots:
	g = sb.relplot(x="time", y=i, kind="line", data=df)
	g.fig.autofmt_xdate()
	fig = g.fig
	fig.savefig(save_dir + "talk_" + i + "_line" + ".png", dpi=fig_dpi)
	plt.cla()
	plt.clf()
	plt.close("all")




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
