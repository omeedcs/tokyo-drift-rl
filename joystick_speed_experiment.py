import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys

path = './dataset/joystick_with_random_imu.csv'
data = pd.read_csv(path)

# get joystick and velocity data
joystick = np.array([eval(i) for i in data["joystick"]])
executed = np.array([eval(i) for i in data["executed"]])

# concatenate joystick and executed data
data = np.concatenate((joystick, executed), axis = 1)

# plot velocities with respect to time
time = []
velocities = []

for i, row in enumerate(data):
    time.append(i)
    velocities.append(row[0])

plt.plot(time, velocities)
plt.xlabel("Time")
plt.ylabel("Velocity")
plt.title("Velocity vs. Time")
plt.show()

    