import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
import math

# CONSTANTS FROM VESC DRIVER LUA. 
# NOTE: These are specific to the UT AUTOmata.
normal_speed = 1.0
turbo_speed = 0.0 # NOTE: we change the value inside of the function.
accel_limit = 6.0
maxTurnRate = 0.25
commandInterval = 1.0 / 20
speed_to_erpm_gain = 5356
speed_to_erpm_offset = 180.0
erpm_speed_limit = 22000
steering_to_servo_gain = -0.9015
steering_to_servo_offset = 0.57
servo_min = 0.05
servo_max = 0.95
wheelbase = 0.324

"""
This function extracts the intertial data from its respective CSV file.
For our use case, we are only worried about "z_1", which is the vertical axis.
"""
def extract_imu_data(filename):
    data_frame = pd.read_csv(filename)
    secs = data_frame["secs"].to_numpy()
    nsecs = data_frame["nsecs"].to_numpy()
    times = secs + nsecs / 1e9 - secs[0]
    imu_angular_vels = -(data_frame["z.1"].to_numpy())
    return (times, imu_angular_vels)

"""
This function extracts the joystick data from its respective CSV file.
"""
def extract_joystick_data(subfolder):
    data_frame = pd.read_csv("./" + subfolder + "/_slash_joystick.csv")
    turbo_speed = 2.0
    secs = data_frame["secs"].to_numpy()
    nsecs = data_frame["nsecs"].to_numpy()
    joystick_times = secs + nsecs / 1e9 - secs[0]
    axes_strings = data_frame["axes"].to_numpy()
    axes = []
    for ax in axes_strings:
        ax = ax[1:-1]
        ax = ax.split(", ")
        ax = [float(a) for a in ax]
        axes.append(ax)
    axes = np.array(axes)
    steer_joystick = -axes[:, 0]
    drive_joystick = -axes[:, 4]
    turbo_mode = axes[:, 2] >= 0.9
    # NOTE: essential line in the creation of training data.
    max_speed = turbo_mode * turbo_speed + (1 - turbo_mode) * normal_speed
    speed = drive_joystick * max_speed
    steering_angle = steer_joystick * maxTurnRate
    last_speed = 0.0
    clipped_speeds = []
    for s in speed:
        smooth_speed = max(s, last_speed - commandInterval*accel_limit)
        smooth_speed = min(smooth_speed, last_speed + commandInterval*accel_limit)
        last_speed = smooth_speed
        erpm = speed_to_erpm_gain * smooth_speed + speed_to_erpm_offset
        erpm_clipped = min(max(erpm, -erpm_speed_limit), erpm_speed_limit)
        clipped_speed = (erpm_clipped - speed_to_erpm_offset) / speed_to_erpm_gain
        clipped_speeds.append(clipped_speed)
    clipped_speeds = np.array(clipped_speeds)
    servo = steering_to_servo_gain * steering_angle + steering_to_servo_offset
    clipped_servo = np.fmin(np.fmax(servo, servo_min), servo_max)
    steering_angle = (clipped_servo - steering_to_servo_offset) / steering_to_servo_gain
    rot_vel = clipped_speeds / wheelbase * np.tan(steering_angle)
    return (joystick_times, clipped_speeds, rot_vel)