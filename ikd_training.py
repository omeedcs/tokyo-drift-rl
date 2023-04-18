# training IKD model with pre-processed MATLAB data and trace the pytorch model to get libtorch model at the end

import numpy as np
import torch
from scipy.io import loadmat
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import sys
import os

class IKDModel(nn.Module):

    def __init__(self, dim_input_command, dim_input_imu):
        super(IKDModel, self).__init__()
        dim_hidden_imu = 256
        dim_output_imu = 2
        self.cmd_dim = dim_input_command
        # self.imu_dim = dim_input_imu
        self.imu = nn.Sequential(
            # nn.Linear(dim_input_imu, dim_hidden_imu),
            # nn.ReLU(),
            nn.Linear(dim_hidden_imu, dim_hidden_imu),
            nn.ReLU(),
            nn.Linear(dim_hidden_imu, dim_output_imu),
            nn.ReLU(),
        )

        dim_hidden = 32
        self.base = nn.Sequential(
            nn.Linear(dim_input_command + dim_output_imu, dim_hidden),
            nn.ReLU(),
            nn.Linear(dim_hidden, dim_hidden),
            nn.ReLU(),
        )
        self.correction = nn.Linear(dim_hidden, 2)

    def forward(self, input):
        input_command, input_imu  = torch.split(input, [self.cmd_dim, self.imu_dim], 1)
        # print(input_imu.size())
        output_imu = self.imu(input_imu)
        # print(output_imu.size())
        # print(input_command.size())
        hidden = torch.cat([input_command, output_imu], -1)
        # print(hidden.size())
        hidden = self.base(hidden)
        # print(hidden.size())
        c = self.correction(hidden)
        c = torch.tanh(c)
        # sys.exit()
        return c


def test(model, data_test, batch_size):
    N_test = data_test.shape[0]
    idx = np.arange(N_test)
    np.random.shuffle(idx)
    input_v = data_test[idx, 2] 
    input_c = data_test[idx, 3]

    # joystick data
    label_v = data_test[idx, 0]
    label_c = data_test[idx, 1] 
    input_imu = data_test[idx, 4:]

    ep_loss = 0
    for i in range(0, N_test, batch_size):
        input_v_ = torch.FloatTensor(input_v[i:min(i+batch_size, N_train)])
        input_c_ = torch.FloatTensor(input_c[i:min(i+batch_size, N_train)])
        label_v_ = torch.FloatTensor(label_v[i:min(i+batch_size, N_train)])
        label_c_ = torch.FloatTensor(label_c[i:min(i+batch_size, N_train)])
        input_imu_ = torch.FloatTensor(input_imu[i:min(i + batch_size, N_train)])

        input_v_ = torch.clamp(input_v_, 0, 6) / 6
        input_c_ = torch.clamp(input_c_, -2, 2) / 2
        label_v_ = torch.clamp(label_v_, 0, 6) / 6
        label_c_ = torch.clamp(label_c_, -2, 2) / 2
        # input_imu_ = input_imu_

        input = torch.cat([input_v_.view(-1, 1), input_c_.view(-1, 1), input_imu_], -1)
        label = torch.cat([label_v_.view(-1, 1), label_c_.view(-1, 1)], -1)

        output = model(input)

        loss = F.mse_loss(output, label)
        ep_loss += loss.item()
    print("[INFO] test loss {:10.4f}".format(ep_loss / (N_test // batch_size)))


if __name__ == '__main__':


    # Training 
    
    data_name = './dataset/ikddata2.csv'
    data = pd.read_csv(data_name)

    joystick = np.array([eval(i) for i in data["joystick"]])
    # realsens = np.array([eval(i) for i in data["executed"]])
    # imu = np.array([eval(i) for i in data["imu"]])

    # imu_mean = np.mean(imu, axis=0)
    # imu_std = np.std(imu, axis=0)
    # imu = (imu - imu_mean) / imu_std

    data = np.concatenate((joystick), axis=1)


    # if os.path.isfile("./ikddata2.pt"):
    #     model = IKDModel(joystick.shape[1], imu.shape[1])
    #     model.load_state_dict(torch.load("./ikddata2.pt"))
    #     input_v = torch.FloatTensor([data[5052, 2]])
    #     input_c = torch.FloatTensor([data[5052, 3]])
    #     label_v = torch.FloatTensor([data[5052, 0]])
    #     label_c = torch.FloatTensor([data[5052, 1]])
    #     input_imu = torch.FloatTensor([data[5052, 4:]])

    #     input_v = torch.clamp(input_v, 0, 6) / 6
    #     input_c = torch.clamp(input_c, -2, 2) / 2
    #     label_v = torch.clamp(label_v, 0, 6) / 6
    #     label_c = torch.clamp(label_c, -2, 2) / 2

    #     input = torch.cat([input_v.view(1, -1), input_c.view(1, -1), input_imu], -1)

    #     traced_script_module = torch.jit.trace(model, input)
    #     traced_script_module.save("ikd_trace.pt")

    # sys.exit()

    N = len(imu)
    N_train = N // 10 * 9
    N_test = N - N_train
    Idx_train = np.random.choice(N, N_train, replace=False)
    mask = np.ones(N, dtype=bool)
    mask[Idx_train] = False
    
    data_train = data[np.invert(mask),]
    data_test = data[mask,]

    # print(joystick.shape[1], imu.shape[1])
    # sys.exit()
    
    model = IKDModel(joystick.shape[1], imu.shape[1])
    opt = torch.optim.Adam(model.parameters(), lr=3e-4, weight_decay=1e-3)
    
    n_ep = 50
    batch_size = 64
    for ep in range(n_ep):
        idx = np.arange(N_train)
        np.random.shuffle(idx)

        # realsense data
        input_v = data_train[idx, 2] 
        input_c = data_train[idx, 3]

        # joystick data
        label_v = data_train[idx, 0]
        label_c = data_train[idx, 1] 

        input_imu = data_train[idx, 4:] # input imu data
    
        ep_loss = 0
    
        for i in range(1, N_train, batch_size):
            input_v_ = torch.FloatTensor(input_v[i:min(i+batch_size, N_train)])
            input_c_ = torch.FloatTensor(input_c[i:min(i+batch_size, N_train)])
            label_v_ = torch.FloatTensor(label_v[i:min(i+batch_size, N_train)])
            label_c_ = torch.FloatTensor(label_c[i:min(i+batch_size, N_train)])
            input_imu_ = torch.FloatTensor(input_imu[i:min(i + batch_size, N_train)])
    
            input_v_ = torch.clamp(input_v_, 0, 6) / 6
            input_c_ = torch.clamp(input_c_, -2, 2) / 2
            label_v_ = torch.clamp(label_v_, 0, 6) / 6
            label_c_ = torch.clamp(label_c_, -2, 2) / 2
            # input_imu_ = input_imu_
    
            input = torch.cat([input_v_.view(-1, 1), input_c_.view(-1, 1), input_imu_], -1)
            label = torch.cat([label_v_.view(-1, 1), label_c_.view(-1, 1)], 1)
            # input = torch.cat([input_v_.view(-1, 1), input_c_.view(-1, 1)], -1)
            output = model(input)
    
            opt.zero_grad()
    
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
