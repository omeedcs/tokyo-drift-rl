import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
from interpolation import get_value_at_time
from extractors import extract_imu_data, extract_joystick_data

"""
This function writes the training data to a CSV file.

NOTE: since we are using vectornav and not realsense, we will not have the GROUNDTRUTH VELOCITY.
We will give it joystick velocity for now and use that as the ground truth velocity.
To do this, we can use extract joystick data function and use the velocity data.

"""
def write_train_data(imu_delay, subfolder):
    
    imu_data = extract_imu_data("./" + subfolder+ "/_slash_vectornav_slash_IMU.csv")
    joystick_times, velocities, rot_vel = extract_joystick_data(subfolder)
    end_time = int(min(imu_data[0][-1], joystick_times[-1]))
    time_points = np.linspace(0, end_time, end_time * 20 + 1)
    joystick = []
    executed = []
    for t in time_points:
        jv = get_value_at_time(t, joystick_times, velocities)
        jav = get_value_at_time(t, joystick_times, rot_vel)
        joystick.append([jv,jav])
        t_av = get_value_at_time(t + imu_delay, imu_data[0], imu_data[1])
        executed.append([t_av])
    training_data = pd.DataFrame()
    training_data["joystick"] = list(joystick)
    training_data["executed"] = list(executed)
    data_file = "./dataset/ikddata2.csv"
    training_data.to_csv(data_file)

    print('Training data extraction and writing complete. The data was written to ' + data_file)
    
def align(subfolder):
    subfolders = [f.name for f in os.scandir("./") if f.is_dir() and f.name != ".git"]
    subfolders = [subfolder]
    for subfolder in subfolders:
        
        print("Beginning data alignment for " + subfolder + ".")

        imu_data = extract_imu_data("./" + subfolder + "/_slash_vectornav_slash_IMU.csv")
        joystick_times, velocities, rot_vel = extract_joystick_data(subfolder)
        end_time = int(min(imu_data[0][-1], joystick_times[-1]))
        time_points = np.linspace(0, end_time, end_time * 20 + 1)
        delay_options = np.linspace(-1, 1, 1001)

        best_cmd_w = []
        for t in time_points:
            best_cmd_w.append(get_value_at_time(t, joystick_times, rot_vel))
        best_cmd_w = np.array(best_cmd_w)

        optimal_delay_imu, optimal_error_imu = 0.0, np.inf
        best_imu_w = []
        for delay in delay_options:
            imu_w = []
            for t in time_points:
                imu_w.append(get_value_at_time(t + delay, imu_data[0], imu_data[1]))
            imu_w = np.array(imu_w)
            error = np.sum((best_cmd_w-imu_w)**2)
            if error < optimal_error_imu:
                optimal_error_imu = error
                optimal_delay_imu = delay
                best_imu_w = imu_w
    return optimal_delay_imu

if __name__ == "__main__":

    # delays recorded:
    # Turn with 1.0 -> .1759
    # Turn with 2.0 -> 0.1779
    # Turn with 3.0 -> .1899
    # Turn with 4.0 -> 0.20599999999999996
    # Turn with 5.0 -> 0.19399
    subfolder = "ikddata2"
    imu_delay = align(subfolder)
    print("imu delay:", imu_delay)
    write_train_data(imu_delay, "ikddata2")