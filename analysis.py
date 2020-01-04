import pickle
import os
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
sns.set(style="darkgrid")


velocity_list_filename = 'velocity_list.dat'

if os.path.exists(velocity_list_filename):
    with open(velocity_list_filename, 'rb') as rfp:
        velocity_list = pickle.load(rfp)

linear_velocity = list()
angular_velocity = list()
linear_acceleration = list()
linear_jerk = list()
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

#calculate jerk
previous_accel = 0
current_accel = 0
timestep = 1

#calculate acceleration
	#acceleration here is timestep derivative of velocity
for i in linear_acceleration:
	current_accel = i
	linear_jerk.append((current_accel - previous_accel)/timestep)
	previous_accel = current_accel


df = pd.DataFrame(dict(time=np.arange(500),
                       value=np.random.randn(500).cumsum()))
g = sns.relplot(x="time", y="value", kind="line", data=df)
g.fig.autofmt_xdate()

#plot linear velocity
x = np.linspace(1, len(linear_velocity), len(linear_velocity))
plt.plot(x, linear_velocity, label="Linear Velocity")
plt.xlabel('Timestep')
plt.ylabel('Velocity')
plt.title('Velocity vs Timestep')
plt.legend()
plt.grid()
plt.show()

#plot linear acceleration
x = np.linspace(1, len(linear_acceleration), len(linear_acceleration))
plt.plot(x, linear_acceleration, label="Linear Acceleration")
plt.xlabel('Timestep')
plt.ylabel('Acceleration')
plt.title('Acceleration vs Timestep')
plt.legend()
plt.grid()
plt.show()

#plot linear jerk
x = np.linspace(1, len(linear_jerk), len(linear_jerk))
plt.plot(x, linear_jerk, label="Linear Jerk")
plt.xlabel('Timestep')
plt.ylabel('Jerk')
plt.title('Jerk vs Timestep')
plt.legend()
plt.grid()
plt.show()

#plot linear jerk
x = np.linspace(1, len(linear_jerk), len(linear_jerk))
plt.plot(x, linear_velocity, label="Linear Velocity")
plt.plot(x, linear_acceleration, label="Linear Acceleration")
plt.plot(x, linear_jerk, label="Linear Jerk")
plt.xlabel('Timestep')
plt.ylabel('Variable')
plt.title('Variable vs Timestep')
plt.legend()
plt.show()


#plot linear velocity and angular velocity
x = np.linspace(1, len(linear_velocity), len(linear_velocity))
plt.plot(x, linear_velocity, label="Linear Velocity")
plt.plot(x, angular_velocity, label="Angular Velocity")
plt.xlabel('Timestep')
plt.ylabel('Velocity')
plt.title('Velocity vs Timestep')
plt.legend()
plt.show()

x = np.linspace(1, len(angular_velocity), len(angular_velocity))
plt.plot(x, angular_velocity)
plt.xlabel('Timestep')
plt.ylabel('Angular Velocity')
plt.title('Angular Velocity vs Timestep')
plt.show()
