# training IKD model with pre-processed MATLAB data and trace the pytorch model to get libtorch model at the end

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
        self.correction = nn.Linear(64, 1)

    def forward(self, input):
        x = self.l1(input)
        x = F.relu(x)
        x = self.l2(x)
        x = F.relu(x)
        x = self.l3(x)
        x = F.relu(x)
        x = self.correction(x)
        x = F.sigmoid(x)
        return x

def test(model, data_test, batch_size):
    N_test = data_test.shape[0]
    idx = np.arange(N_test)
    np.random.shuffle(idx)
    
    # joystick velocity
    joystick_v = data_train[idx, 0]

    # joystick angular velocity
    joystick_av = data_train[idx, 1] 

    # ground truth angular velocity
    true_av = data_train[idx, 2:]

    # current epoch loss
    ep_loss = 0

    print("[INFO] testing...")
    for i in range(1, N_test, batch_size):       
    
        joystick_v_tens = torch.FloatTensor(joystick_v[i:min(i+batch_size, N_train)])
        joystick_av_tens = torch.FloatTensor(joystick_av[i:min(i+batch_size, N_train)])
        true_av_tens = torch.FloatTensor(true_av[i:min(i+batch_size, N_train)])

        joystick_v_tens = torch.clamp(joystick_v_tens, 0, 6) / 6
        joystick_av_tens = torch.clamp(joystick_av_tens, -2, 2) / 2

        input = torch.cat([joystick_v_tens.view(-1, 1), true_av_tens], -1)
        label = joystick_av_tens.view(-1, 1)
        output = model(input)
        loss = F.mse_loss(output, label)

        loss.backward()
        opt.step()
        ep_loss += loss.item()
    print("[INFO] test loss {:10.4f}".format(ep_loss / (N_test // batch_size)))


if __name__ == '__main__':

    # NOTE: TRAINING 
    data_name = './dataset/ikddata2.csv'
    data = pd.read_csv(data_name)
    joystick = np.array([eval(i) for i in data["joystick"]])
    executed = np.array([eval(i) for i in data["executed"]])
    data = np.concatenate((joystick, executed), axis = 1)
    
    N = data.shape[0]
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
    batch_size = 64
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
    
        for i in range(1, N_train, batch_size):       
        
            joystick_v_tens = torch.FloatTensor(joystick_v[i:min(i+batch_size, N_train)])
            joystick_av_tens = torch.FloatTensor(joystick_av[i:min(i+batch_size, N_train)])
            true_av_tens = torch.FloatTensor(true_av[i:min(i+batch_size, N_train)])

            joystick_v_tens = torch.clamp(joystick_v_tens, 0, 6) / 6
            joystick_av_tens = torch.clamp(joystick_av_tens, -2, 2) / 2

            input = torch.cat([joystick_v_tens.view(-1, 1), true_av_tens], -1)
            label = joystick_av_tens.view(-1, 1)
            output = model(input)
            loss = F.mse_loss(output, label)

            loss.backward()
            opt.step()
            ep_loss += loss.item()

        print("[INFO] epoch {} | loss {:10.4f}".format(ep, ep_loss / (N_train // batch_size)))
    
        test(model, data_test, batch_size)
        torch.save(model.state_dict(), "ikddata2.pt")
    print('done')


    # trace the pytorch model to libtorch model

    # model = IKDModel(2, 600)
    # model.load_state_dict(torch.load("training_backyard_ackermann_imu_enconder.pt"))
    # data = loadmat('/home/xuesu/Desktop/offroadnav_data/backyard_training/ackermann/training_backyard_ackermann_imu_ALL.mat')
    # data = np.array(data['all_data'])

    # imu_mean = np.mean(data[:, 3:], axis=0)  # angular mean 0.01572 0.11090 0.24443 accel mean 0.07645 0.21097 9.56521
    # imu_std = np.std(data[:, 3:], axis=0)  # angular std 0.69783 0.43708 1.64938 accel std 1.66148 4.50831 3.77231
    # data[:, 3:] = (data[:, 3:] - imu_mean) / imu_std

    # input_v_ = torch.FloatTensor([data[9268, 0]])
    # input_c_ = torch.FloatTensor([data[9268, 2]])
    # label_c_ = torch.FloatTensor([data[9268, 1]])
    # input_imu_ = torch.FloatTensor([data[9268, 3:]])

    # input_v_ = torch.clamp(input_v_, 0, 3) / 3
    # input_c_ = torch.clamp(input_c_, -2, 2) / 2
    # label_c_ = torch.clamp(label_c_, -2, 2) / 2
    # input_imu_ = input_imu_

    # input = torch.cat([input_v_.view(-1, 1), input_c_.view(-1, 1), input_imu_], -1)

    # traced_script_module = torch.jit.trace(model, input)
    # traced_script_module.save("traced_training_backyard_ackermann_imu_encoder_final_experiments.pt")
    # output = traced_script_module(input)
    # print(input_v_ * 3)
    # print(input_c_ * 2)
    # print(label_c_ * 2)
    # print(output * 2)



    # NOTE: MODEL TRACING FOR LIBTORCH INPUT:
    
    # if os.path.isfile("./ikddata2.pt"):
    #     model = IKDModel(1, 1)
    #     model.load_state_dict(torch.load("./ikddata2.pt"))
    #     label_v = torch.FloatTensor([data[5052, 0]])
    #     label_c = torch.FloatTensor([data[5052, 1]])
    #     input_imu = torch.FloatTensor([data[5052, 2]])

    #     label_v = torch.clamp(label_v, 0, 6) / 6
    #     label_c = torch.clamp(label_c, -2, 2) / 2
    #     input_imu = torch.clamp(input_imu, -2, 2) / 2

    #     input = torch.cat([label_v.view(-1, 1), input_imu.view(-1, 1)], -1)

    #     traced_script_module = torch.jit.trace(model, input)
    #     traced_script_module.save("ikd_trace.pt")

    # sys.exit()

# Load in the trained model
model = IKDModel(2, 1)
model.load_state_dict(torch.load("./ikddata2.pt"))
joystick_v = 0.7938474416732788
joystick_av = 1.7420551901185741
joystick_v_norm = torch.clamp(torch.FloatTensor([joystick_v]), 0, 6) / 6
joystick_av_norm = torch.clamp(torch.FloatTensor([joystick_av]), -2, 2) / 2
joystick_v_norm = joystick_v_norm.view(-1, 1)
joystick_av_norm = joystick_av_norm.view(-1, 1)
input = torch.cat([joystick_v_norm, joystick_av_norm], -1)
output = model(input)
joystick_av_pred = output * 2 
print(joystick_av_pred)
