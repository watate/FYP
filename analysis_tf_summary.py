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


#get data
for e in tf.compat.v1.train.summary_iterator('ddpg_summary/20200106-214225/events.out.tfevents.1578318156.ubuntu'):
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


crash = 0
get_target = 0
time_out = 0
for i in result:
	if i == 1: time_out = time_out + 1;
	if i == 2: crash = crash + 1;
	if i == 3: get_target = get_target + 1;


print("Crash: {}".format(crash))
print("Get Target: {}".format(get_target))
print("Time Out: {}".format(time_out))


#Generating the DataFrame
df = pd.DataFrame(dict(time=np.arange(len(linear_velocity)),
                       linear_velocity=linear_velocity,
						linear_acceleration=linear_acceleration,
						linear_jerk=linear_jerk,
						angular_velocity=angular_velocity,
						angular_acceleration=angular_acceleration,
						angular_jerk=angular_jerk,
						))

with open('analysis_dataframe.dat','wb') as wfp:
	pickle.dump(df, wfp)