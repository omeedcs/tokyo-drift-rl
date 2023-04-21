import numpy as np
import torch
from scipy.io import loadmat
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import sys
import os

# https://github.com/ut-amrl/ikd/blob/main/ikd_training.py

"""
This is the class for the Inverse Kinodynamic Model.

The model has 2 inputs: 
- The velocity of the controller, which we denote as v.
- The angular velocity from the IMU, which we denote as av'. This is the ground truth.
The model has 1 output:
- The velocity of the controller + the Angular Velocity of the controller, which we denoted as (v, av).

This neural network is a simple neural network with 2 hidden layers.
"""

class IKDModel(nn.Module):
    def __init__(self, dim_input, dim_output):
        super(IKDModel, self).__init__()
        self.l1 = nn.Linear(dim_input, 64)
        self.l2 = nn.Linear(64, 128)
        self.l3 = nn.Linear(128, 64)
        self.l4 = nn.Linear(64, 32)
        self.correction = nn.Linear(32, dim_output)

    def forward(self, input):
        x = self.l1(input)
        x = F.relu(x)
        x = self.l2(x)
        x = F.relu(x)
        x = self.l3(x)
        x = F.relu(x)
        x = self.l4(x)
        x = F.relu(x)
        x = self.correction(x)
        return x

if __name__ == '__main__':

    # NOTE: TRAINING 
    data_name = './dataset/ikddata2.csv'
    data = pd.read_csv(data_name)
    joystick = np.array([eval(i) for i in data["joystick"]])
    executed = np.array([eval(i) for i in data["executed"]])

    data = np.concatenate((joystick, executed), axis = 1)
    N = len(executed)
    N_train = N // 10 * 9
    N_test = N - N_train
    idx_train = np.random.choice(N, N_train, replace = False)
    mask = np.ones(N, dtype=bool)
    mask[idx_train] = False

    data_train = data[np.invert(mask),]
    data_test = data[mask,]
    model = IKDModel(2, 1)
    opt = torch.optim.Adam(model.parameters(), lr = 1e-4, weight_decay = 1e-3)
    
    n_ep = 100
    batch_size = 1

    for ep in range(n_ep):

        idx = np.arange(N_train)
        np.random.shuffle(idx)

        # joystick velocity
        joystick_v = data_train[idx, 0]

        # joystick angular velocity
        joystick_av = data_train[idx, 1] 

        # ground truth angular velocity
        true_av = data_train[idx, 2:]

        # current epoch loss
        ep_loss = 0
    
        for i in range(0, N_train, batch_size):       
        
            joystick_v_tens = torch.FloatTensor(joystick_v[i:min(i + batch_size, N_train)])
            joystick_av_tens = torch.FloatTensor(joystick_av[i:min(i + batch_size, N_train)])
            true_av_tens = torch.FloatTensor(true_av[i:min(i + batch_size, N_train)])

            # normalizes the joystick velocity.
            joystick_v_tens = torch.clamp(joystick_v_tens, 0, 6) / 6
            joystick_av_tens = torch.clamp(joystick_av_tens, -4, 4) / 4

            jv = joystick_v_tens.view(-1, 1)
            jav = joystick_av_tens.view(-1, 1)
            tav = true_av_tens
            input = torch.cat([jv, tav], -1)
            output = model(input)
            opt.zero_grad()
            loss = F.mse_loss(output, jav.view(-1, 1))
            loss.backward()
            opt.step()
            ep_loss += loss.item()

        print("[INFO] epoch {} | loss {:10.4f}".format(ep, ep_loss / (N_train // batch_size)))

        # test(model, data_test, batch_size)
        torch.save(model.state_dict(), "ikddata2.pt")
    print('done')

# # Load in the trained model
model = IKDModel(2, 1)
model.load_state_dict(torch.load("./ikddata2.pt"))
joystick_v = 0.7938474416732788
joystick_av = 0.6256250954112949
joystick_v_norm = torch.clamp(torch.FloatTensor([joystick_v]), 0, 6) / 6
joystick_av_norm = torch.clamp(torch.FloatTensor([joystick_av]), -4, 4) / 4
joystick_v_norm = joystick_v_norm.view(-1, 1)
joystick_av_norm = joystick_av_norm.view(-1, 1)
input = torch.cat([joystick_v_norm, joystick_av_norm], -1)
normalized_output = torch.clamp(torch.FloatTensor([1.6527914337671066]), -4, 4) / 4
print(input)
output = model(input)
print("---")
print("Our Output:", output.item())
print("Our Input: ", input)
print("Normalized Correct Output:", normalized_output)
print("Non-Normalized Correct Output:", 1.6527914337671066)
print("---")

# 1,"[0.7938474416732788, 0.6256250954112949]",[1.6527914337671066]
