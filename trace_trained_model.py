# This script traces the trained pytorch model to libtorch model and saves it to a file.
# This is neccesary because the pytorch model is not compatible with the C++ API on the car.

import numpy as np
import torch
import pandas as pd
from ikd_training import IKDModel

model = IKDModel(2, 1)
model.load_state_dict(torch.load("./ikddata2.pt"))
data_name = './dataset/ikddata2.csv'
data = pd.read_csv(data_name)
joystick = np.array([eval(i) for i in data["joystick"]])
executed = np.array([eval(i) for i in data["executed"]])
data = np.concatenate((joystick, executed), axis = 1)

joystick_v_tens = torch.FloatTensor([data[2560, 0]])
joystick_av_tens = torch.FloatTensor([data[2560, 1]])
true_av_tens = torch.FloatTensor([data[2560, 2]])

joystick_v_tens = torch.clamp(joystick_v_tens, 0, 6) / 6
joystick_av_tens = torch.clamp(joystick_av_tens, -4, 4) / 4
true_av_tens = torch.clamp(true_av_tens, -4, 4) / 4

jv = joystick_v_tens.view(-1, 1)
jav = joystick_av_tens.view(-1, 1)
tav = true_av_tens.view(-1, 1)

input = torch.cat([jv, tav], dim = -1)

traced_script_module = torch.jit.trace(model, input)
traced_script_module.save("traced_ikd_model.pt")
output = traced_script_module(input)
print(traced_script_module)
print(joystick_v_tens * 6)
print(joystick_av_tens * 4)
print(true_av_tens * 4)
print(output * 4)