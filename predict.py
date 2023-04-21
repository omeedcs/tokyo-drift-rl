import numpy as np
import torch
from scipy.io import loadmat
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import sys
import os
from ikd_training import IKDModel

# # Load in the trained model
model = IKDModel(2, 1)
model.load_state_dict(torch.load("./ikddata2.pt"))
joystick_v = 0.6765323975471114
joystick_av = -1.2319081520198218
# normalize the input
joystick_v = joystick_v / 6
joystick_av = joystick_av / 4
# joystick tensor
# input tensor
input = torch.FloatTensor([joystick_v, joystick_av])
with torch.no_grad():
    output = model(input)
print("---")
print("Our Input:", torch.FloatTensor([joystick_v, joystick_av]))
print("Our velocity:", joystick_v * 6)
print("Our Output (denormalized):", output.item() * 4)
print("Non-Normalized Correct Output:", -0.5046003772907106)
print("---")
# 2335,"[3.7750160876753567, -2.9750612977387205]",[-1.946162323413568]
# 1047,"[1.0, 0.7880923494476428]",[0.9036172882863241]
# 53,"[1.0, -0.7149554779268497]",[-0.9208256471273581]
# 75,"[1.0, 0.7880923494476428]",[]
# 2424,"[0.6765323975471114, -0.5046003772907106]",[-1.2319081520198218]
