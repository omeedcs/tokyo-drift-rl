import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sys

# read the CSV file
data = pd.read_csv('./dataset/drift_with_cone.csv')

joystick = np.array([eval(i) for i in data["joystick"]])
executed = np.array([eval(i) for i in data["executed"]])
data = np.concatenate((joystick, executed), axis = 1)

linear_velocity = data[:, 0]
angular_velocity = data[:, 1]
executed = data[:, 2]

print(executed[0])

# generate a time array based on the number of records
time = range(len(joystick))

# plot the linear velocity and angular velocity against time
fig, ax = plt.subplots(figsize=(8, 8))
ax.plot(time, angular_velocity, label='Commanded Angular Velocity')
ax.plot(time, executed, label='True Angular Velocity Velocity')
ax.legend()
ax.set_xlabel('Time')
ax.set_ylabel('Angular Velocity')
plt.show()