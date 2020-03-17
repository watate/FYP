#This script applies exponential moving average to our data (to smooth it out and see the long-term trends)

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

#Set EMA span
EMA_span = 1000

######################### LOADING DATAFRAME DATA #########################
#Obtain raw values from previous dataframe
reward = df["reward"].ewm(span=EMA_span, adjust=True).mean()
q_max = df["q_max"].ewm(span=EMA_span, adjust=True).mean()
distance = df["distance"].ewm(span=EMA_span, adjust=True).mean()
steps = df["steps"].ewm(span=EMA_span, adjust=True).mean()
result = df["result"]
pid_rate = df["pid_rate"]
linear_jerk = df["linear_jerk"].ewm(span=EMA_span, adjust=True).mean()
angular_jerk = df["angular_jerk"].ewm(span=EMA_span, adjust=True).mean()


policy_usage = [1 - i for i in pid_rate]

######################### CALCULATE SUCCESS RATE #########################
#Convert result into success/failure
success_rate = [1 if i == 3 else 0 for i in result]

data_temp = {"Placeholder": policy_usage}
df_temp = pd.DataFrame(data_temp)
policy_usage = df_temp["Placeholder"].ewm(span=1000, adjust=True).mean()

data_temp = {"Placeholder": success_rate}
df_temp = pd.DataFrame(data_temp)
success_rate = df_temp["Placeholder"].ewm(span=1000, adjust=True).mean()

data = {"Time": np.arange(len(reward)),
		"Reward": reward,
		"Qmax": q_max,
		"Distance": distance,
		"Steps": steps,
		"Policy Usage": policy_usage,
		"Success Rate": success_rate,
		"Total Linear Jerk": linear_jerk,
		"Total Angular Jerk": angular_jerk}

#Generate second dataframe for averaged results
df2 = pd.DataFrame(data, columns=["Time", "Reward", "Qmax",
	"Distance", "Steps", "Policy Usage", "Success Rate",
	 "Total Linear Jerk", "Total Angular Jerk"])

######################## SAVE DATA #########################
with open('dataframe_smoothed/' + shortname + '_smooth.dat','wb') as wfp:
	pickle.dump(df2, wfp)
