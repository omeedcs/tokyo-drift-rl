import numpy as np
import torch
from scipy.io import loadmat
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import sys
import os
from ikd_model import IKDModel
import matplotlib.pyplot as plt

# NOTE: shell taken from the inverse kinodynamic training code:
# https://github.com/ut-amrl/ikd/blob/main/ikd_training.py

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
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
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr = 1e-5, weight_decay = 1e-3)
    n_ep = 50
    training_loss = []
    testing_loss = []
    epochs = []
    batch_size = 32

    for ep in range(n_ep):
        
        idx = np.arange(N_train)
        np.random.shuffle(idx)

        # joystick velocity
        joystick_v = data_train[idx, 0]

        # joystick angular velocity
        joystick_av = data_train[idx, 1] 

        # ground truth angular velocity
        true_av = data_train[idx, 2]

        # current epoch loss
        ep_loss = 0

        model.train()
    
        for i in range(0, N_train, batch_size):    

            opt.zero_grad() 

            joystick_v_tens = torch.FloatTensor(joystick_v[i:min(i + batch_size, N_train)])
            joystick_av_tens = torch.FloatTensor(joystick_av[i:min(i + batch_size, N_train)])
            true_av_tens = torch.FloatTensor(true_av[i:min(i + batch_size, N_train)])
            
            jv = joystick_v_tens.view(-1, 1)
            jav = joystick_av_tens.view(-1, 1)
            tav = true_av_tens.view(-1, 1)
            
            input = torch.cat([jv, tav], dim = -1)

            output = model(input)
            loss = F.mse_loss(output, jav)
            loss.backward()
            opt.step()
            ep_loss += loss.item()
        training_loss.append(ep_loss / (N_train // batch_size)) 
        epochs.append(ep)
        print("[INFO] epoch {} | loss {:10.4f}".format(ep, ep_loss / (N_train // batch_size)))

        model.eval()
        test_loss = 0
        for i in range(0, N_test, batch_size):    

            opt.zero_grad() 

            joystick_v_tens = torch.FloatTensor(joystick_v[i:min(i + batch_size, N_train)])
            joystick_av_tens = torch.FloatTensor(joystick_av[i:min(i + batch_size, N_train)])
            true_av_tens = torch.FloatTensor(true_av[i:min(i + batch_size, N_train)])
            
            jv = joystick_v_tens.view(-1, 1)
            jav = joystick_av_tens.view(-1, 1)
            tav = true_av_tens.view(-1, 1)
            
            input = torch.cat([jv, tav], dim = -1)

            output = model(input)
            loss = F.mse_loss(output, jav)
            loss.backward()
            opt.step()
            test_loss += loss.item()
        testing_loss.append(test_loss / (N_test // batch_size))
        print("[INFO] epoch {} | loss {:10.4f}".format(ep, test_loss / (N_test // batch_size)))
        torch.save(model.state_dict(), "ikddata2.pt")
    print("[INFO] Training complete. Model saved as 'ikddata2.pt")
    print("[INFO] Plotting loss values...")
    # plot the linear velocity and angular velocity against time
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.plot(epochs, testing_loss, label='Testing Loss')
    ax.plot(epochs, training_loss, label='Training Loss')
    ax.legend()
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Loss')
    plt.show()
