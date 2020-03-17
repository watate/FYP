

#This script applies exponential moving average to our data (to smooth it out and see the long-term trends)

import pickle
import os
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sb
import pandas as pd

##############################################################################
#Prepare and choose datasets from which to generate graphs #ADD_HERE
#filename = "20200301-074856_34000c_dataframe.dat"

######################### SIMPLE ################################
#filename = "20200302-044536_24000s_dataframe.dat"
#filename = "20200311-004451_VSmooth_Simple_dataframe.dat"
#filename = "20200309-000445_Jerk_Simple_dataframe.dat"
#shortname = "Jerk_Simple"

######################### COMPLEX ################################
#filename = "20200301-074856_34000c_dataframe.dat"
#filename = "20200311-203436_VSmooth_Complex_dataframe.dat"
filename = "20200309-214742_Jerk_Complex_dataframe.dat"
shortname = "Jerk_Complex"

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


policy_usage = [1 - i for i in pid_rate]

######################### CALCULATE SUCCESS RATE #########################
#Convert result into success/failure
success_rate = [1 if i == 3 else 0 for i in result]

data_temp = {"Placeholder": reward}
df_temp = pd.DataFrame(data_temp)
reward = df_temp["Placeholder"].ewm(span=EMA_span, adjust=True).mean()

data_temp = {"Placeholder": q_max}
df_temp = pd.DataFrame(data_temp)
q_max = df_temp["Placeholder"].ewm(span=EMA_span, adjust=True).mean()

data_temp = {"Placeholder": distance}
df_temp = pd.DataFrame(data_temp)
distance = df_temp["Placeholder"].ewm(span=EMA_span, adjust=True).mean()

data_temp = {"Placeholder": steps}
df_temp = pd.DataFrame(data_temp)
steps = df_temp["Placeholder"].ewm(span=EMA_span, adjust=True).mean()

data_temp = {"Placeholder": policy_usage}
df_temp = pd.DataFrame(data_temp)
policy_usage = df_temp["Placeholder"].ewm(span=EMA_span, adjust=True).mean()

data_temp = {"Placeholder": success_rate}
df_temp = pd.DataFrame(data_temp)
success_rate = df_temp["Placeholder"].ewm(span=EMA_span, adjust=True).mean()

data = {"Time": np.arange(len(reward)),
		"Reward": reward,
		"Qmax": q_max,
		"Distance": distance,
		"Steps": steps,
		"Policy Usage": policy_usage,
		"Success Rate": success_rate}

#Generate second dataframe for averaged results
df2 = pd.DataFrame(data, columns=["Time", "Reward", "Qmax",
	"Distance", "Steps", "Policy Usage", "Success Rate"])

######################## SAVE DATA #########################
with open('dataframe_smoothed/' + shortname + '_smooth.dat','wb') as wfp:
	pickle.dump(df2, wfp)
