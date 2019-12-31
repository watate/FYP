import pickle
import os
import matplotlib.pyplot as plt
import numpy as np

velocity_list_filename = 'velocity_list.dat'

if os.path.exists(velocity_list_filename):
    with open(velocity_list_filename, 'rb') as rfp:
        velocity_list = pickle.load(rfp)

linear_velocity = list()
angular_velocity = list()
noise = list()
difference = list()

for i in velocity_list:
	linear_velocity.append(i[0])
	angular_velocity.append(i[1])
	noise.append(i[2])
	#difference.append(i[0] - i[2])


#plot linear velocity
x = np.linspace(1, len(linear_velocity), len(linear_velocity))
plt.plot(x, linear_velocity)
plt.xlabel('Timestep')
plt.ylabel('Linear Velocity')
plt.title('Linear Velocity vs Timestep')
plt.show()

x = np.linspace(1, len(angular_velocity), len(angular_velocity))
plt.plot(x, angular_velocity)
plt.xlabel('Timestep')
plt.ylabel('Angular Velocity')
plt.title('Angular Velocity vs Timestep')
plt.show()

x = np.linspace(1, len(noise), len(noise))
plt.plot(x, noise)
plt.xlabel('Timestep')
plt.ylabel('Noise')
plt.title('Noise vs Timestep')
plt.show()

x = np.linspace(1, len(linear_velocity), len(linear_velocity))
plt.plot(x, difference)
plt.xlabel('Timestep')
plt.ylabel('Difference')
plt.title('Difference vs Timestep')
plt.show()