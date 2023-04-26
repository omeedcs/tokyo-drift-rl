import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

path = './dataset/ikddata2.csv'
data = pd.read_csv(path)

# get joystick and velocity data
joystick = np.array([eval(i) for i in data["joystick"]])
executed = np.array([eval(i) for i in data["executed"]])

# concatenate joystick and executed data
data = np.concatenate((joystick, executed), axis = 1)

# plot velocities with respect to time
time = []
velocities = []
angular_velocities = []
true_angular_velocities = []
curvatures = []

for i, row in enumerate(data):
    time.append(i)
    velocities.append(row[0])
    angular_velocities.append(row[1])
    true_angular_velocities.append(row[2])

    velocity = row[0]
    angular_velocity = row[1]
    true_angular_velocity = row[2]

    # get the curvature
    if velocity == 0.0:
        curvatures.append(0.0)
    else:
        curvature = angular_velocity / velocity
        curvatures.append(curvature)


plt.plot(time, angular_velocities)
plt.xlabel("Time")
plt.ylabel("Angular Velocity")
plt.title("Angular Velocity vs. Time")
plt.show()


plt.plot(time, true_angular_velocities)
plt.xlabel("Time")
plt.ylabel("True Angular Velocity")
plt.title("True Angular Velocity vs. Time")
plt.show()

plt.plot(time, curvatures)
plt.xlabel("Time")
plt.ylabel("Curvature")
plt.title("Curvature vs. Time")
plt.show()


print(max(curvatures))
print(min(curvatures))
# we can control number of bins with num!
bin_ranges = np.linspace(min(curvatures), max(curvatures), num=8)
hist, bin_edges = np.histogram(curvatures, bins=bin_ranges)

# Plot the histogram
plt.figure(figsize=(10, 5))
plt.bar(bin_edges[:-1], hist, width=np.diff(bin_edges), edgecolor="k", align="edge")
plt.xlabel("Curvature")
plt.ylabel("Frequency")
plt.title("Curvature Histogram")
plt.show()


# plot the linear velocity and angular velocity against time
fig, ax = plt.subplots(figsize=(8, 8))
ax.plot(time, angular_velocities, label='Commanded Angular Velocity')
ax.plot(time, true_angular_velocities, label='True Angular Velocity')
ax.legend()
ax.set_xlabel('Time')
ax.set_ylabel('Angular Velocity')
plt.show()