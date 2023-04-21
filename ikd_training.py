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
        self.l1 = nn.Linear(2, 32)
        self.l2 = nn.Linear(32, 32)
        self.correction = nn.Linear(32, dim_output)

    def forward(self, input):

        x = F.relu(self.l1(input))
        x = F.relu(self.l2(x))
        x = self.correction(x)
        x = torch.tanh(x)
        return x
    
if __name__ == '__main__':

    # NOTE: TRAINING 
    data_name = './dataset/ikddata2.csv'
    data = pd.read_csv(data_name)
    joystick = np.array([eval(i) for i in data["joystick"]])
    executed = np.array([eval(i) for i in data["executed"]])
    # print("joystick", joystick.shape)
    # print("executed", executed.shape)
    data = np.concatenate((joystick, executed), axis = 1)

    
    # NOTE: do we need?
    # imu_mean = np.mean(data[:,2:], axis=0) # angular mean 0.01572 0.11090 0.24443 accel mean 0.07645 0.21097 9.56521
    # imu_std = np.std(data[:, 2:], axis=0) # angular std 0.69783 0.43708 1.64938 accel std 1.66148 4.50831 3.77231
    # data[:, 2:] = (data[:, 2:] - imu_mean) / imu_std
    N = len(executed)
    N_train = N // 10 * 9
    N_test = N - N_train
    idx_train = np.random.choice(N, N_train, replace = False)
    mask = np.ones(N, dtype=bool)
    mask[idx_train] = False
    data_train = data[np.invert(mask),]
    data_test = data[mask,]

    model = IKDModel(2, 1)
    opt = torch.optim.Adam(model.parameters(), lr = 3e-4, weight_decay = 1e-3)
    
    n_ep = 100
    batch_size = 32

    for ep in range(n_ep):
        
        idx = np.arange(N_train)
        np.random.shuffle(idx)

        # joystick velocity
        joystick_v = data_train[idx, 0]
        # print("joystick_v", joystick_v)
        # print("joystick_v.shape", joystick_v.shape)

        # joystick angular velocity
        joystick_av = data_train[idx, 1] 
        # print("joystick_av", joystick_av)
        # print("joystick_av.shape", joystick_av.shape)

        # ground truth angular velocity
        true_av = data_train[idx, 2]
        # print("true_av", true_av)
        # print("true_av.shape", true_av.shape)

        # current epoch loss
        ep_loss = 0
    
        for i in range(0, N_train, batch_size):    

            opt.zero_grad() 

            joystick_v_tens = torch.FloatTensor(joystick_v[i:min(i + batch_size, N_train)])
            joystick_av_tens = torch.FloatTensor(joystick_av[i:min(i + batch_size, N_train)])
            true_av_tens = torch.FloatTensor(true_av[i:min(i + batch_size, N_train)])
            
            joystick_v_tens = torch.clamp(joystick_v_tens, 0, 6) / 6
            joystick_av_tens = torch.clamp(joystick_av_tens, -4, 4) / 4
            true_av_tens = torch.clamp(true_av_tens, -4, 4) / 4

            jv = joystick_v_tens.view(-1, 1)
            jav = joystick_av_tens.view(-1, 1)
            tav = true_av_tens.view(-1, 1)
            
            input = torch.cat([jv, tav], dim = -1)

            output = model(input)
            loss = F.mse_loss(output, jav)
            loss.backward()
            opt.step()
            ep_loss += loss.item()
        
        print("[INFO] epoch {} | loss {:10.4f}".format(ep, ep_loss / (N_train // batch_size)))

        # test(model, data_test, batch_size)
        torch.save(model.state_dict(), "ikddata2.pt")
    print('done')


