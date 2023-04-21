import numpy as np
import torch
from scipy.io import loadmat
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import sys
import os

class IKDModel(nn.Module):
    def __init__(self, dim_input, dim_output):
        super(IKDModel, self).__init__()
        self.l1 = nn.Linear(dim_input, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, 64)
        self.correction = nn.Linear(64, dim_output)


    def forward(self, input):
        x = F.relu(self.l1(input))
        x = F.relu(self.l2(x))
        x = F.relu(self.l3(x))
        x = self.correction(x)
        return x
    
# # Load in the trained model
model = IKDModel(2, 1)
model.load_state_dict(torch.load("./ikddata2.pt"))
joystick_v = 1.0
joystick_av = 0.7880923494476428
# normalize the input
joystick_v = joystick_v / 6
joystick_av = joystick_av / 4
with torch.no_grad():
    output = model(torch.FloatTensor([joystick_v, joystick_av]))
print("---")
print("Our Input:", torch.FloatTensor([joystick_v, joystick_av]))
print("Our velocity:", joystick_v * 6)
print("Our Output (denormalized):", output.item() * 4)
print("Non-Normalized Correct Output:", 0.9036172882863241)
print("---")

# 1047,"[1.0, 0.7880923494476428]",[0.9036172882863241]
