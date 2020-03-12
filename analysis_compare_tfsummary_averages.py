#This script is like analysis_compare_tfsummary but it generates dataframe for averaged values

import pickle
import os
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sb
import pandas as pd

##############################################################################
#Prepare and choose datasets from which to generate graphs #ADD_HERE
filename = "20200301-074856_34000c_dataframe.dat"
shortname = "Normal_Complex"

#Loading the DataFrame generated from preprocess (analysis_tf_summary.py)
#ADD_HERE
if os.path.exists(filename):
    with open(filename, 'rb') as rfp:
        df = pickle.load(rfp)

######################### AVERAGING THE DATA #########################
#Obtain raw values from previous dataframe
reward = df["reward"]
q_max = df["q_max"]
distance = df["distance"]
steps = df["steps"]
pid_rate = df["pid_rate"]
result = df["result"]
linear_jerk = df["linear_jerk"]
angular_jerk = df["angular_jerk"]

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

#Convert result into success/failure
result = [1 if i == 3 else 0 for i in result]
#Separate the groups. The last slice will be fine with less than 100 numbers.
groups = [result[x:x+100] for x in range(0, len(result), 100)]
#calculate mean
success_rate = [sum(group)/len(group) for group in groups]

Do the same for jerk
groups = [linear_jerk[x:x+100] for x in range(0, len(linear_jerk), 100)]
average_total_linear_jerk = [sum(group)/len(group) for group in groups]
groups = [angular_jerk[x:x+100] for x in range(0, len(angular_jerk), 100)]
average_total_angular_jerk = [sum(group)/len(group) for group in groups]


data = {"Time": np.arange(len(average_performance)),
		"Average Reward": average_reward,
		"Average Performance": average_performance,
		"Average Distance": average_distance,
		"Average Steps": average_steps,
		"Average PID Rate": average_pid_rate,
		"Policy Usage": policy_usage,
		"Success Rate": success_rate,
		"Average Total Linear Jerk": average_total_linear_jerk,
		"Average Total Angular Jerk": average_total_angular_jerk}

#Generate second dataframe for averaged results
df2 = pd.DataFrame(data, columns=["Time", "Average Reward", "Average Performance",
	"Average Distance", "Average Steps", "Average PID Rate", "Policy Usage", "Success Rate"
	 "Average Total Linear Jerk", "Average Total Angular Jerk"])


######################## SAVE DATA #########################
with open('dataframe_averages/' + shortname + '_average.dat','wb') as wfp:
	pickle.dump(df2, wfp)