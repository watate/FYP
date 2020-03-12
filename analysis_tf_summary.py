#Extracts data from tf.summary file and generates a dataframe
	#Note: Qmax is the average maximum q-value of an episode

import tensorflow as tf
import pickle
import os
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sb
import pandas as pd

#list
reward = list()
q_max = list()
pid_rate = list()
distance = list()
result = list()
steps = list()
linear_jerk = list()
angular_jerk = list()


#folderdate = "20200226-093050" #Blank World
#folderdate = "20200226-170933" #Simple World (after training in b)
#folderdate = "20200227-220051" #Complex World (after training in s and b)
#folderdate = "20200302-044536" #24000s (Model 1)
#folderdate = "20200301-074856" #34000c (Model 1)
folderdate = "20200311-004451" #Model 2 Simple World
#folderdate = "20200310-201253" #Model 3 Blank World
#folderdate = "20200309-000445" #?? Model 3 Simple World
#folderdate = "20200309-214742" # Model 3 Complex World
shortname = "VSmooth_Simple"

#get data
for e in tf.compat.v1.train.summary_iterator('ddpg_summary/' + folderdate + '/events.out.tfevents.1583912693.ubuntu'):
	for v in e.summary.value:
		if v.tag == 'Reward':
			reward.append(v.simple_value)
		if v.tag == 'Qmax':
			q_max.append(v.simple_value)
		if v.tag == 'PIDrate':
			pid_rate.append(v.simple_value)
		if v.tag == 'Distance':
			distance.append(v.simple_value)
		if v.tag == 'Result':
			result.append(v.simple_value)
		if v.tag == 'Steps':
			steps.append(v.simple_value)
		if v.tag == 'Total Linear Jerk':
			linear_jerk.append(v.simple_value)
		if v.tag == 'Total Angular Jerk':
			angular_jerk.append(v.simple_value)


# crash = 0
# get_target = 0
# time_out = 0
# for i in result:
# 	if i == 1: time_out = time_out + 1;
# 	if i == 2: crash = crash + 1;
# 	if i == 3: get_target = get_target + 1;


# print("Crash: {}".format(crash))
# print("Get Target: {}".format(get_target))
# print("Time Out: {}".format(time_out))

#Generating the DataFrame
df = pd.DataFrame(dict(time=np.arange(len(reward)),
						reward=reward,
					   q_max=q_max,
					   pid_rate=pid_rate,
					   distance=distance,
					   result=result,
					   steps=steps,
					   linear_jerk=linear_jerk,
					   angular_jerk=angular_jerk,
						))

with open(folderdate + '_' + shortname + '_dataframe.dat','wb') as wfp:
	pickle.dump(df, wfp)