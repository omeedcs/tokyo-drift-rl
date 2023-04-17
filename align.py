import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
import math
from math_helpers import get_value_at_time, euler_from_quaternion, subt_poses
from extractors import extract_imu_data, extract_joystick_data, extract_tracking_cam_data, extract_tracking_cam_odom, extract_vesc_odom, find_start_and_end_time

# NOTE: changed to match UT Automata car.
normal_speed = 1.0
turbo_speed = 6.0
accel_limit = 6.0
maxTurnRate = 0.25
commandInterval = 1.0 / 20
speed_to_erpm_gain = 5356.0
speed_to_erpm_offset = 180.0
erpm_speed_limit = 22000.0
steering_to_servo_gain = -0.9015 
steering_to_servo_offset = 0.57 
servo_min = 0.05 
servo_max = 0.95 
wheelbase = 0.324

steer_joystick_idx = 0
drive_joystick_idx = 3

# given a time, return how many seconds it is from the start of the bag
# s is the number of seconds to look back, default is 1 second meaning 40 data points
def get_imu_data_at_time(time, imu_time, imu_accel, imu_gyro, s = 1):
    imu_hz = 40
    window_size = s * imu_hz

    # find imu_time that is closest to time
    left = 0
    right = imu_time.size-1
    mid = 0
    while right-left > 1:
        mid = (left+right)//2
        if (time > imu_time[mid]):
            left = mid
        else:
            right = mid
    return np.concatenate([imu_accel[mid-window_size:mid].flatten(), imu_gyro[mid-window_size:mid].flatten()])

def write_train_data(imu_delay, cam_delay, odom_delay, subfolder):
    
    imu_data = extract_imu_data("./" + subfolder+ "/_slash_vectornav_slash_IMU.csv")
    joystick_data = extract_joystick_data(subfolder)
    imu_accel = imu_data[2]
    imu_gyro = imu_data[3]
    start, end, starti, endi = find_start_and_end_time(joystick_data[0], joystick_data[1])
    print("start: ", start, " end: ", end)
    time_points = np.linspace(start + 2, end, 8000)
    joystick = []
    true_disp = []
    odom_disp = []
    executed = []
    imu_accel_gyro = []
    for t in time_points:
        # joystick velocity and curvature
        jv = get_value_at_time(t, joystick_data[0], joystick_data[1])
        jc = get_value_at_time(t, joystick_data[0], joystick_data[3])
        joystick.append([jv,jc])
        l = list(get_imu_data_at_time(t + imu_delay, imu_data[0], imu_accel, imu_gyro))
        imu_accel_gyro.append(l)
    
    training_data = pd.DataFrame()
    training_data["joystick"] = list(joystick)
    # training_data["executed"] = list(executed) #probably correct
    training_data["imu"] = list(imu_accel_gyro)
    data_file = "./dataset/ikddata2.csv"
    training_data.to_csv(data_file)

    print('training data written to ' + data_file)
    
def align(subfolder):
    subfolders = [f.name for f in os.scandir("./") if f.is_dir() and f.name != ".git"]
    subfolders = [subfolder]
    for subfolder in subfolders:
        print("Aligning data for", subfolder)
        
        # extract the commanded velocity & angular velocity
        joystick_data = extract_joystick_data(subfolder)
        
        # extract IMU data
        imu_data = extract_imu_data("./"+subfolder+"/_slash_vectornav_slash_IMU.csv")

        # Compute delay
        end_time = int(min(joystick_data[0][-1], imu_data[0][-1]))
        print(end_time)
        time_points = np.linspace(0, end_time, 8000)
        delay_options = np.linspace(-1, 1, 1001)

        best_cmd_w = []
        for t in time_points:
            best_cmd_w.append(get_value_at_time(t, joystick_data[0], joystick_data[2]))
        best_cmd_w = np.array(best_cmd_w)

        optimal_delay_imu, optimal_error_imu = 0.0, np.inf
        best_imu_w = []
        for delay in delay_options:
            imu_w = []
            for t in time_points:
                imu_w.append(get_value_at_time(t+delay, imu_data[0], imu_data[1]))
            imu_w = np.array(imu_w)
            error = np.sum((best_cmd_w-imu_w)**2)
            if error < optimal_error_imu:
                optimal_error_imu = error
                optimal_delay_imu = delay
                best_imu_w = imu_w

    print(subfolder+" imu delay:", optimal_delay_imu)

    plt.show()  

    return optimal_delay_imu


if __name__ == "__main__":
    subfolder = "ikddata2"

    if subfolder == "ikddata2":
        imu_delay = 0.176
        cam_delay = 0.172
        odom_delay = -0.588

    # NOTE: fixed align to be purely IMU/Joystick (omeed)
    # imu_delay = align(subfolder)
    # print("imu delay:", imu_delay)

    write_train_data(imu_delay,cam_delay,odom_delay,"ikddata2")   