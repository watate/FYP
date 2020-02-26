#### Data Preprocessing

import pickle
import os
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sb
import pandas as pd

velocity_list_filename = 'velocity_list.dat'

if os.path.exists(velocity_list_filename):
    with open(velocity_list_filename, 'rb') as rfp:
        velocity_list = pickle.load(rfp)

linear_velocity = list()
angular_velocity = list()
linear_acceleration = list()
linear_jerk = list()
angular_acceleration = list()
angular_jerk = list()
#noise = list()
#difference = list()

starting = 30
j = 0

for i in velocity_list:
	if j > starting:
		linear_velocity.append(i[0])
		angular_velocity.append(i[1])
		#difference.append(i[0] - i[2])
	j = j + 1

previous_vel = 0
current_vel = 0
timestep = 1

#calculate acceleration
	#acceleration here is timestep derivative of velocity
for i in linear_velocity:
	current_vel = i
	linear_acceleration.append((current_vel - previous_vel)/timestep)
	previous_vel = current_vel

previous_vel = 0
current_vel = 0
timestep = 1

#calculate acceleration
	#acceleration here is timestep derivative of velocity
for i in angular_velocity:
	current_vel = i
	angular_acceleration.append((current_vel - previous_vel)/timestep)
	previous_vel = current_vel

#calculate jerk
previous_accel = 0
current_accel = 0
timestep = 1

#calculate jerk
	#jerk here is timestep derivative of acceleration
for i in linear_acceleration:
	current_accel = i
	linear_jerk.append((current_accel - previous_accel)/timestep)
	previous_accel = current_accel

#calculate jerk
previous_accel = 0
current_accel = 0
timestep = 1

#calculate jerk
	#jerk here is timestep derivative of acceleration
for i in angular_acceleration:
	current_accel = i
	angular_jerk.append((current_accel - previous_accel)/timestep)
	previous_accel = current_accel


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