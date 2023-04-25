
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.io import loadmat
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from ikd_training import IKDModel
# import plt
import matplotlib.pyplot as plt

"""
Extracting the data out of the folder.
"""
turn_with_1 = './simple-turning-data/turn_with_1.0.csv'
turn_with_2 = './simple-turning-data/turn_with_2.0.csv'
turn_with_3 = './simple-turning-data/turn_with_3.0.csv'
turn_with_4 = './simple-turning-data/turn_with_4.0.csv'
turn_with_5 = './simple-turning-data/turn_with_5.0.csv'

turn_with_1 = pd.read_csv(turn_with_1)
turn_with_2 = pd.read_csv(turn_with_2)
turn_with_3 = pd.read_csv(turn_with_3)
turn_with_4 = pd.read_csv(turn_with_4)
turn_with_5 = pd.read_csv(turn_with_5)

# get joystick and executed data
joystick_1 = np.array([eval(i) for i in turn_with_1["joystick"]])
executed_1 = np.array([eval(i) for i in turn_with_1["executed"]])

joystick_2 = np.array([eval(i) for i in turn_with_2["joystick"]])
executed_2 = np.array([eval(i) for i in turn_with_2["executed"]])

joystick_3 = np.array([eval(i) for i in turn_with_3["joystick"]])
executed_3 = np.array([eval(i) for i in turn_with_3["executed"]])

joystick_4 = np.array([eval(i) for i in turn_with_4["joystick"]])
executed_4 = np.array([eval(i) for i in turn_with_4["executed"]])

joystick_5 = np.array([eval(i) for i in turn_with_5["joystick"]])
executed_5 = np.array([eval(i) for i in turn_with_5["executed"]])

# concatenate joystick and executed data
data_1 = np.concatenate((joystick_1, executed_1), axis = 1)
data_2 = np.concatenate((joystick_2, executed_2), axis = 1)
data_3 = np.concatenate((joystick_3, executed_3), axis = 1)
data_4 = np.concatenate((joystick_4, executed_4), axis = 1)
data_5 = np.concatenate((joystick_5, executed_5), axis = 1)


"""
Section #1: Plots the angular velocities and times.
"""

time = []
angular_velocity = []
true_angular_velocity = []

for i, row in enumerate(data_1):
    time.append(i)
    angular_velocity.append(row[1])
    true_angular_velocity.append(row[2])


# plot the linear velocity and angular velocity against time
fig, ax = plt.subplots(figsize=(8, 8))
ax.plot(time, angular_velocity, label='Commanded Angular Velocity')
ax.plot(time, true_angular_velocity, label='True Angular Velocity Velocity')
ax.legend()
ax.set_xlabel('Time')
ax.set_ylabel('Angular Velocity')
plt.show()

time = []
angular_velocity = []
true_angular_velocity = []

for i, row in enumerate(data_2):
    time.append(i)
    angular_velocity.append(row[1])
    true_angular_velocity.append(row[2])

# plot the linear velocity and angular velocity against time
fig, ax = plt.subplots(figsize=(8, 8))
ax.plot(time, angular_velocity, label='Commanded Angular Velocity')
ax.plot(time, true_angular_velocity, label='True Angular Velocity Velocity')
ax.legend()
ax.set_xlabel('Time')
ax.set_ylabel('Angular Velocity')
plt.show()

time = []
angular_velocity = []
true_angular_velocity = []

for i, row in enumerate(data_3):
    time.append(i)
    angular_velocity.append(row[1])
    true_angular_velocity.append(row[2])

# plot the linear velocity and angular velocity against time
fig, ax = plt.subplots(figsize=(8, 8))
ax.plot(time, angular_velocity, label='Commanded Angular Velocity')
ax.plot(time, true_angular_velocity, label='True Angular Velocity Velocity')
ax.legend()
ax.set_xlabel('Time')
ax.set_ylabel('Angular Velocity')
plt.show()

time = []
angular_velocity = []
true_angular_velocity = []

for i, row in enumerate(data_4):
    time.append(i)
    angular_velocity.append(row[1])
    true_angular_velocity.append(row[2])

# plot the linear velocity and angular velocity against time
fig, ax = plt.subplots(figsize=(8, 8))
ax.plot(time, angular_velocity, label='Commanded Angular Velocity')
ax.plot(time, true_angular_velocity, label='True Angular Velocity')
ax.legend()
ax.set_xlabel('Time')
ax.set_ylabel('Angular Velocity')
plt.show()

time = []
angular_velocity = []
true_angular_velocity = []

for i, row in enumerate(data_5):
    time.append(i)
    angular_velocity.append(row[1])
    true_angular_velocity.append(row[2])

# plot the linear velocity and angular velocity against time
fig, ax = plt.subplots(figsize=(8, 8))
ax.plot(time, angular_velocity, label='Commanded Angular Velocity')
ax.plot(time, true_angular_velocity, label='True Angular Velocity Velocity')
ax.legend()
ax.set_xlabel('Time')
ax.set_ylabel('Angular Velocity')
plt.show()

"""
Section #2: Histogram bin the curvatures to see the diversity of the data.
"""
curvatures = []

for i, row in enumerate(data_1):
    velocity = row[0]
    angular_velocity = row[1]
    true_angular_velocity = row[2]

    # get the curvature
    if velocity == 0.0:
        curvatures.append(0.0)
    else:
        curvature = angular_velocity / velocity
        curvatures.append(curvature)

# we can control number of bins with num!
bin_ranges = np.linspace(min(curvatures), max(curvatures), num=4)  
hist, bin_edges = np.histogram(curvatures, bins=bin_ranges)

# Plot the histogram
plt.figure(figsize=(10, 5))
plt.bar(bin_edges[:-1], hist, width=np.diff(bin_edges), edgecolor="k", align="edge")
plt.xlabel("Curvature")
plt.ylabel("Frequency")
plt.title("Curvature Histogram of 1.0 m/s")
plt.show()

curvatures = []

for i, row in enumerate(data_2):
    velocity = row[0]
    angular_velocity = row[1]
    true_angular_velocity = row[2]

    # get the curvature
    if velocity == 0.0:
        curvatures.append(0.0)
    else:
        curvature = angular_velocity / velocity
        curvatures.append(curvature)

# we can control number of bins with num!
bin_ranges = np.linspace(min(curvatures), max(curvatures), num=4)
hist, bin_edges = np.histogram(curvatures, bins=bin_ranges)

# Plot the histogram
plt.figure(figsize=(10, 5))
plt.bar(bin_edges[:-1], hist, width=np.diff(bin_edges), edgecolor="k", align="edge")
plt.xlabel("Curvature")
plt.ylabel("Frequency")
plt.title("Curvature Histogram of 2 m/s")
plt.show()

curvatures = []

for i, row in enumerate(data_3):
    velocity = row[0]
    angular_velocity = row[1]
    true_angular_velocity = row[2]

    # get the curvature
    if velocity == 0.0:
        curvatures.append(0.0)
    else:
        curvature = angular_velocity / velocity
        curvatures.append(curvature)

# we can control number of bins with num!
bin_ranges = np.linspace(min(curvatures), max(curvatures), num=4)
hist, bin_edges = np.histogram(curvatures, bins=bin_ranges)

# Plot the histogram
plt.figure(figsize=(10, 5))
plt.bar(bin_edges[:-1], hist, width=np.diff(bin_edges), edgecolor="k", align="edge")
plt.xlabel("Curvature")
plt.ylabel("Frequency")
plt.title("Curvature Histogram of 3 m/s")
plt.show()

curvatures = []

for i, row in enumerate(data_4):
    velocity = row[0]
    angular_velocity = row[1]
    true_angular_velocity = row[2]

    # get the curvature
    if velocity == 0.0:
        curvatures.append(0.0)
    else:
        curvature = angular_velocity / velocity
        curvatures.append(curvature)

# we can control number of bins with num!
bin_ranges = np.linspace(min(curvatures), max(curvatures), num=4)
hist, bin_edges = np.histogram(curvatures, bins=bin_ranges)

# Plot the histogram
plt.figure(figsize=(10, 5))
plt.bar(bin_edges[:-1], hist, width=np.diff(bin_edges), edgecolor="k", align="edge")
plt.xlabel("Curvature")
plt.ylabel("Frequency")
plt.title("Curvature Histogram of 4 m/s")
plt.show()

curvatures = []

for i, row in enumerate(data_5):
    velocity = row[0]
    angular_velocity = row[1]
    true_angular_velocity = row[2]

    # get the curvature
    
    # get the curvature
    if velocity == 0.0:
        curvatures.append(0.0)
    else:
        curvature = angular_velocity / velocity
        curvatures.append(curvature)

# we can control number of bins with num!
bin_ranges = np.linspace(min(curvatures), max(curvatures), num=4)
hist, bin_edges = np.histogram(curvatures, bins=bin_ranges)

# Plot the histogram
plt.figure(figsize=(10, 5))
plt.bar(bin_edges[:-1], hist, width=np.diff(bin_edges), edgecolor="k", align="edge")
plt.xlabel("Curvature")
plt.ylabel("Frequency")
plt.title("Curvature Histogram of 5 m/s")
plt.show()

"""
Merging and binning curvatures...
"""
turn_with_1 = './simple-turning-data/turn_with_1.0.csv'
turn_with_2 = './simple-turning-data/turn_with_2.0.csv'
turn_with_3 = './simple-turning-data/turn_with_3.0.csv'
turn_with_4 = './simple-turning-data/turn_with_4.0.csv'
turn_with_5 = './simple-turning-data/turn_with_5.0.csv'

turn_with_1 = pd.read_csv(turn_with_1)
turn_with_2 = pd.read_csv(turn_with_2)
turn_with_3 = pd.read_csv(turn_with_3)
turn_with_4 = pd.read_csv(turn_with_4)
turn_with_5 = pd.read_csv(turn_with_5)

# rewrite concat 
data = pd.concat([turn_with_1, turn_with_2, turn_with_3, turn_with_4, turn_with_5], axis = 0)
data = data.reset_index(drop = True)
data = data.drop(columns=["Unnamed: 0"])

# save merged csv
data.to_csv("./simple-turning-data/merged-turning-data.csv")

path = "./simple-turning-data/merged-turning-data.csv"

data = pd.read_csv(path)

# get joystick and velocity data
joystick = np.array([eval(i) for i in data["joystick"]])
executed = np.array([eval(i) for i in data["executed"]])

# concatenate joystick and executed data
data = np.concatenate((joystick, executed), axis = 1)

curvatures = []

for i, row in enumerate(data):
    velocity = row[0]
    angular_velocity = row[1]
    true_angular_velocity = row[2]

    # get the curvature
    if velocity == 0.0:
        curvatures.append(0.0)
    else:
        curvature = angular_velocity / velocity
        curvatures.append(curvature)

# we can control number of bins with num!
bin_ranges = np.linspace(min(curvatures), max(curvatures), num=6)
hist, bin_edges = np.histogram(curvatures, bins=bin_ranges)

# Plot the histogram
plt.figure(figsize=(10, 5))
plt.bar(bin_edges[:-1], hist, width=np.diff(bin_edges), edgecolor="k", align="edge")
plt.xlabel("Curvature")
plt.ylabel("Frequency")
plt.title("Curvature Histogram of 1.0 to 5.0 m/s")
plt.show()

# bin velocities
velocities = []

for i, row in enumerate(data):
    velocity = row[0]
    angular_velocity = row[1]
    true_angular_velocity = row[2]

    velocities.append(velocity)

# we can control number of bins with num!
bin_ranges = np.linspace(min(velocities), max(velocities), num=10)
hist, bin_edges = np.histogram(velocities, bins=bin_ranges)

# Plot the histogram
plt.figure(figsize=(10, 5))
plt.bar(bin_edges[:-1], hist, width=np.diff(bin_edges), edgecolor="k", align="edge")
plt.xlabel("Velocity")
plt.ylabel("Frequency")
plt.title("Velocity Histogram of 1.0 to 5.0 m/s")
plt.show()