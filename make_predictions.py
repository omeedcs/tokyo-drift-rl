import numpy as np
import torch
from scipy.io import loadmat
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from ikd_training import IKDModel
import matplotlib.pyplot as plt

model = IKDModel(2, 1)
model.load_state_dict(torch.load("./ikddata2.pt"))
data_name = './dataset/loose_ccw.csv'
data = pd.read_csv(data_name)
joystick = np.array([eval(i) for i in data["joystick"]])
executed = np.array([eval(i) for i in data["executed"]])
data = np.concatenate((joystick, executed), axis = 1)

actual_av = []
corrected_av = []
time = []

# iterate over all training data, and make predictions accordingly.
for i, row in enumerate(data):
    time.append(i)
    v = row[0]
    av = row[1]
    true_av = row[2]
    input = torch.FloatTensor([v, av])
    output = model(input)
    corrected_av.append(true_av * 2)
    actual_av.append(true_av)

fig, ax = plt.subplots()
ax.plot(time, corrected_av, label='Corrected Angular Velocity')
ax.plot(time, actual_av, label='Actual Angular Velocity')
ax.legend()
ax.set_title("Loose CCW")
ax.set_xlabel('Time')
ax.set_ylabel('Angular Velocities')
plt.show()

# model = IKDModel(2, 1)
# model.load_state_dict(torch.load("./ikddata2.pt"))
# data_name = './dataset/loose_cw.csv'
# data = pd.read_csv(data_name)
# joystick = np.array([eval(i) for i in data["joystick"]])
# executed = np.array([eval(i) for i in data["executed"]])
# data = np.concatenate((joystick, executed), axis = 1)

# actual_av = []
# corrected_av = []
# time = []

# # iterate over all training data, and make predictions accordingly.
# for i, row in enumerate(data):
#     time.append(i)
#     v = row[0]
#     av = row[1]
#     true_av = row[2]
#     input = torch.FloatTensor([v, av])
#     output = model(input)
#     corrected_av.append(output.item())
#     actual_av.append(true_av)


# fig, ax = plt.subplots()
# ax.plot(time, corrected_av, label='Corrected Angular Velocity')
# ax.plot(time, actual_av, label='Actual Angular Velocity')
# ax.legend()
# ax.set_xlabel('Time')
# ax.set_ylabel('Angular Velocities')
# ax.set_title("Loose CW")
# plt.show()

# model = IKDModel(2, 1)
# model.load_state_dict(torch.load("./ikddata2.pt"))
# data_name = './dataset/tight_cw.csv'
# data = pd.read_csv(data_name)
# joystick = np.array([eval(i) for i in data["joystick"]])
# executed = np.array([eval(i) for i in data["executed"]])
# data = np.concatenate((joystick, executed), axis = 1)

# actual_av = []
# corrected_av = []
# time = []

# # iterate over all training data, and make predictions accordingly.
# for i, row in enumerate(data):
#     time.append(i)
#     v = row[0]
#     av = row[1]
#     true_av = row[2]
#     input = torch.FloatTensor([v, av])
#     output = model(input)
#     corrected_av.append(output.item())
#     actual_av.append(true_av)


# fig, ax = plt.subplots()
# ax.plot(time, corrected_av, label='Corrected Angular Velocity')
# ax.plot(time, actual_av, label='Actual Angular Velocity')
# ax.legend()
# ax.set_xlabel('Time')
# ax.set_ylabel('Angular Velocities')
# ax.set_title("Tight CW")
# plt.show()

# model = IKDModel(2, 1)
# model.load_state_dict(torch.load("./ikddata2.pt"))
# data_name = './dataset/tight_ccw.csv'
# data = pd.read_csv(data_name)
# joystick = np.array([eval(i) for i in data["joystick"]])
# executed = np.array([eval(i) for i in data["executed"]])
# data = np.concatenate((joystick, executed), axis = 1)

# actual_av = []
# corrected_av = []
# time = []

# # iterate over all training data, and make predictions accordingly.
# for i, row in enumerate(data):
#     time.append(i)
#     v = row[0]
#     av = row[1]
#     true_av = row[2]
#     input = torch.FloatTensor([v, av])
#     output = model(input)
#     corrected_av.append(output.item())
#     actual_av.append(true_av)


# fig, ax = plt.subplots()
# ax.plot(time, corrected_av, label='Corrected Angular Velocity')
# ax.plot(time, actual_av, label='Actual Angular Velocity')
# ax.legend()
# ax.set_xlabel('Time')
# ax.set_ylabel('Angular Velocities')
# ax.set_title("Tight CCW")
# plt.show()

