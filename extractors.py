import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
import math
from math_helpers import euler_from_quaternion

# NOTE: changed to match UT Automata car.
normal_speed = 1.0
turbo_speed = 6.0
accel_limit = 6.0
maxTurnRate = 0.25
commandInterval = 1.0 / 20
speed_to_erpm_gain = 5356
speed_to_erpm_offset = 180.0
erpm_speed_limit = 22000
steering_to_servo_gain = -.9015 
steering_to_servo_offset = 0.57 
servo_min = 0.05 
servo_max = 0.95 
wheelbase = 0.324

steer_joystick_idx = 0
drive_joystick_idx = 3

def extract_vesc_odom(subfolder):
    data_frame = pd.read_csv("./mat_data/"+subfolder+"/_slash_odom.csv")
    nsecs = data_frame["rosbagTimestamp"].to_numpy()
    # nsecs = data_frame["nsecs"].to_numpy()
    times = (nsecs - nsecs[0]) / 1e9
    
    # you need to find differences in each timestep
    x_pos = data_frame["x"].to_numpy()
    y_pos = data_frame["y"].to_numpy()
    roll, pitch, yaw = euler_from_quaternion(data_frame['x.1'].to_numpy(), data_frame['y.1'].to_numpy(),data_frame['z.1'].to_numpy(),data_frame['w'].to_numpy());

    # each difference should be associated with the timestamp right before it, it should correspond with the joystick command that caused this displacement to happen
    # x_pos = np.diff(x_pos)
    # y_pos = np.diff(y_pos)
    theta = np.diff(yaw)
    diff_time = np.diff(times)
    # times = times[:-1]

    # print([i for i in ang_vel if i != 0])

    ang_vel = theta / diff_time;
    # ang_vel = np.concatenate

    return (times, x_pos, y_pos, yaw, ang_vel)


# gets the position and orientation     
def extract_tracking_cam_odom(subfolder):
    data_frame = pd.read_csv("./mat_data/"+subfolder+"/_slash_camera_slash_odom_slash_sample.csv")

    secs = data_frame["secs"].to_numpy()
    nsecs = data_frame["nsecs"].to_numpy()
    times = secs + nsecs / 1e9 - secs[0]
    
    # you need to find differences in each timestep
    x_pos = data_frame["x"].to_numpy()
    y_pos = data_frame["y"].to_numpy()
    _, _, yaw = euler_from_quaternion(data_frame['x.1'].to_numpy(), data_frame['y.1'].to_numpy(),data_frame['z.1'].to_numpy(),data_frame['w'].to_numpy());

    # each difference should be associated with the timestamp right before it, it should correspond with the joystick command that caused this displacement to happen
    # x_pos = np.diff(x_pos)
    # y_pos = np.diff(y_pos)
    # theta = np.diff(yaw)
    # times = times[:-1]
    durations = np.diff(times)

    return (times, x_pos, y_pos, yaw, durations)

    # for ikd slam, you do the same input to the model where you encode imu into 

def extract_tracking_cam_data(subfolder):
    data_frame = pd.read_csv("./mat_data/"+subfolder+"/_slash_camera_slash_odom_slash_sample.csv")

    secs = data_frame["secs"].to_numpy()
    nsecs = data_frame["nsecs"].to_numpy()
    times = secs + nsecs / 1e9 - secs[0]

    x_vel = data_frame["x.2"].to_numpy()
    y_vel = data_frame["y.2"].to_numpy()
    velocities = (x_vel**2 + y_vel**2)**0.5
    velocities = np.array([velocities[i] if x_vel[i] > 0 else -velocities[i] for i in range(velocities.size)])
    angular_vels = data_frame["z.3"].to_numpy()
    # angular_vels = np.array([angular_vels[i] if velocities[i] > 0.05 else 0 for i in range(angular_vels.size)])
    realsense_curvature = angular_vels / velocities

    # plt.plot(times, angular_vels, label='w')
    # plt.plot(times, velocities, label = 'v')
    # plt.legend()
    # plt.show()

    return (times, velocities, angular_vels, x_vel, y_vel, realsense_curvature)

def find_start_and_end_time(time, vels):


    start = -1
    end = -1
    starti, endi = 0, 0
    for i in range(vels.size):
        if vels[i] > 0.4:
            start = time[i]
            starti = i
            break
    for i in range(vels.size-1, -1, -1):
        if vels[i] > 0.4:
            end = time[i]
            endi = i
            break
    # print(end-start)
    return start, end, starti, endi

# def integrate(t, y):
#     res = [0.0000000001]
#     for i in range(1, t.size):
#         res.append(np.trapz(y[:i], x=t[:i]))

#     return np.array(res)

def extract_imu_data(filename):

    data_frame = pd.read_csv(filename)

    secs = data_frame["secs"].to_numpy()
    nsecs = data_frame["nsecs"].to_numpy()
    times = secs + nsecs / 1e9 - secs[0]

    # you need xyz of accelerometer and gyroscope
    imu_angular_vels = -(data_frame["z.1"].to_numpy())

    imu_accel = np.column_stack((data_frame["x.2"].to_numpy(), data_frame["y.2"].to_numpy(), data_frame["z.2"].to_numpy()))
    imu_gyro = np.column_stack((data_frame["x.1"].to_numpy(), data_frame["y.1"].to_numpy(), data_frame["z.1"].to_numpy()))
    
    return (times, imu_angular_vels, imu_accel, imu_gyro)


def extract_joystick_data(subfolder):
    data_frame = pd.read_csv("./"+subfolder+"/_slash_joystick.csv")

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

    # print(len([axe[0] for axe in axes if axe[0] != 0]))
    # print(len([axe[0] for axe in axes if axe[0] == 0]))

    steer_joystick = axes[:, steer_joystick_idx]
    drive_joystick = axes[:, drive_joystick_idx]
    max_speed = normal_speed
    speed = drive_joystick*max_speed                # array of all the speeds
    steering_angle = steer_joystick*maxTurnRate     # array of all the steering angles?
    
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
    # print(steering_angle)
    # atan(wheelbase*curvature) == steering_angle
    curvature = np.tan(steering_angle) / wheelbase
    rot_vel = clipped_speeds / wheelbase * np.tan(steering_angle)
    # print(steering_angle)
    # print(joystick_times)
    # print(clipped_speeds)
    # print(rot_vel)
    # print(curvature)
    return (joystick_times, clipped_speeds, rot_vel, curvature)
