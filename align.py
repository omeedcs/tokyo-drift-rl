import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
import math
from math_helpers import get_value_at_time, euler_from_quaternion, subt_poses
from extractors import extract_imu_data, extract_joystick_data, extract_tracking_cam_data, extract_tracking_cam_odom, extract_vesc_odom, find_start_and_end_time

normal_speed = 6.0
turbo_speed = 0.0
accel_limit = 6.0
maxTurnRate = 0.25#done
commandInterval = 1.0/20#done
speed_to_erpm_gain = 4790#done
speed_to_erpm_offset = 0#done
erpm_speed_limit = 30000#done
steering_to_servo_gain = -1.205 #done
steering_to_servo_offset = 0.53 #done
servo_min = 0.05#done
servo_max = 0.95#done
wheelbase = 0.480#done

steer_joystick_idx = 0
drive_joystick_idx = 3

# given a time, return how many 
def get_imu_data_at_time(time, imu_time, imu_accel, imu_gyro, s=1):
    imu_hz = 40
    window_size = s*imu_hz

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

def process_odom_data(times, x, y, theta):
    # caculate pairwise things
    times = times[:-1]
    dx = []
    dy = []
    dtheta = []
    for i in range(len(times)):
        x2, y2, theta2 = subt_poses(x[i-1], y[i-1], theta[i-1], x[i], y[i], theta[i])
        dx.append(x2)
        dy.append(y2)
        dtheta.append(theta2)

    return times, np.array(dx), np.array(dy), np.array(dtheta)


def write_train_data(imu_delay, cam_delay, odom_delay, subfolder):
    cam_odom = extract_tracking_cam_odom(subfolder)
    odom = extract_vesc_odom(subfolder)
    imu_data = extract_imu_data("./mat_data/"+subfolder+"/_slash_vectornav_slash_IMU.csv")
    joystick_data = extract_joystick_data(subfolder)
    cam_data = extract_tracking_cam_data(subfolder)
    imu_accel = imu_data[2]
    imu_gyro = imu_data[3]

    cam_odom = process_odom_data(cam_odom[0], cam_odom[1], cam_odom[2], cam_odom[3])
    odom = process_odom_data(odom[0], odom[1], odom[2], odom[3])

    start, end,starti, endi = find_start_and_end_time(joystick_data[0], joystick_data[1])

    time_points = np.linspace(start+2, end, 8000)
    joystick = []
    true_disp = []
    odom_disp = []
    executed = []
    imu_accel_gyro = []
    for t in time_points:
        jv = get_value_at_time(t, joystick_data[0], joystick_data[1])
        jc = get_value_at_time(t, joystick_data[0], joystick_data[3])
        joystick.append([jv,jc])
        ev = get_value_at_time(t+cam_delay, cam_data[0], cam_data[1])
        ec = get_value_at_time(t+cam_delay, cam_data[0], cam_data[5])
        if abs(ec) > abs(jc*1.5):
            ec = joystick[-1][1]
        executed.append([ev, ec])
        dx = get_value_at_time(t+cam_delay, cam_odom[0], cam_odom[1])
        dy = get_value_at_time(t+cam_delay, cam_odom[0], cam_odom[2])
        dtheta = get_value_at_time(t+cam_delay, cam_odom[0], cam_odom[3])
        true_disp.append([dx, dy, dtheta])
        dx = get_value_at_time(t+odom_delay, odom[0], odom[1])
        dy = get_value_at_time(t+odom_delay, odom[0], odom[2])
        dtheta = get_value_at_time(t+odom_delay, odom[0], odom[3])
        odom_disp.append([dx, dy, dtheta])
        imu_accel_gyro.append(list(get_imu_data_at_time(t+imu_delay, imu_data[0], imu_accel, imu_gyro)))
        # print(imu_accel_gyro[0])

    training_data = pd.DataFrame()
    training_data["odom_disp"] = list(odom_disp)
    training_data["true_disp"] = list(true_disp)
    training_data["joystick"] = list(joystick)
    training_data["executed"] = list(executed) #probably correct
    training_data["imu"] = list(imu_accel_gyro)
    data_file = "./dataset/ikddata2.csv"
    training_data.to_csv(data_file)

    print('training data written to ' + data_file)


def visualize_odom(subfolder, cam_delay, odom_delay):
    cam_data = extract_tracking_cam_odom(subfolder)
    odom_data = extract_vesc_odom(subfolder)
    joystick_data = extract_joystick_data(subfolder)

    start, end,starti, endi = find_start_and_end_time(joystick_data[0], joystick_data[1])
    
    # print(start)
    # print(end)
    # print(odom_data[0][0])
    # print(odom_data[0][-1])
    # print(cam_data[0][0])
    # print(cam_data[0][-1])
    
    time_points = np.linspace(start, end, 8000)
    odomx = []
    odomy = []
    odomtheta = []
    camx = []
    camy = []
    camtheta = []
    for t in time_points:
        odomx.append(get_value_at_time(t+odom_delay, odom_data[0], odom_data[1]))
        odomy.append(get_value_at_time(t+odom_delay, odom_data[0], odom_data[2]))
        odomtheta.append(get_value_at_time(t+odom_delay, odom_data[0], odom_data[3]))
        camx.append(get_value_at_time(t+cam_delay, cam_data[0], cam_data[1]))
        camy.append(get_value_at_time(t+cam_delay, cam_data[0], cam_data[2]))
        camtheta.append(get_value_at_time(t+cam_delay, cam_data[0], cam_data[3]))


    odomx = np.array(odomx)
    odomy = np.array(odomy)
    odomtheta = np.array(odomtheta)
    camtheta = np.array(camtheta)
    camx = np.array(camx)
    camy = np.array(camy)

    odomx -= odomx[0]
    odomy -= odomy[0]
    camx -= camx[0]
    camy -= camy[0]

    costheta = np.cos(-odomtheta[0])
    sintheta = np.cos(-odomtheta[0])
    odomx = odomx*costheta - odomy*sintheta
    odomy = odomy*costheta + odomx*sintheta

    costheta = np.cos(-camtheta[0])
    sintheta = np.cos(-camtheta[0])
    camx = camx*costheta - camy*sintheta
    camy = camy*costheta + camx*sintheta

    odomtheta += -odomtheta[0]
    camtheta += -camtheta[0]

    # plt.scatter(cam_data[1], cam_data[2])
    # plt.scatter(odom_data[1], odom_data[2])
    plt.scatter(odomx[:500], odomy[:500], label="odom")
    plt.scatter(camx[:500], camy[:500], label="cam")
    plt.legend()
    plt.show()

def align(subfolder):
    subfolders = [f.name for f in os.scandir("./mat_data/") if f.is_dir() and f.name != ".git"]
    subfolders = [subfolder]
    for subfolder in subfolders:
        #Extract commanded velocity & angular velocity
        joystick_data = extract_joystick_data(subfolder)

        #Extract tracking cam linear and angular velocity data
        tracking_cam_data = extract_tracking_cam_data(subfolder)
            
        # extract IMU data
        imu_data = extract_imu_data("./mat_data/"+subfolder+"/_slash_vectornav_slash_IMU.csv")

        # odom data
        odom_data = extract_vesc_odom(subfolder) # odom_data[4] is ang vel
        odom_ang_vel = [i if abs(i) < 10 else 0 for i in odom_data[4]]

        # Compute delay
        end_time = int(min(joystick_data[0][-1], min(imu_data[0][-1], tracking_cam_data[0][-1])))
        time_points = np.linspace(0, end_time, 8000)
        delay_options = np.linspace(-1, 1, 1001)

        best_cmd_w = []
        for t in time_points:
            best_cmd_w.append(get_value_at_time(t, joystick_data[0], joystick_data[2]))
        best_cmd_w = np.array(best_cmd_w)

        optimal_delay_cam, optimal_error_cam = 0.0, np.inf
        best_w = []
        for delay in delay_options:
            w = []
            for t in time_points:
                w.append(get_value_at_time(t+delay, tracking_cam_data[0], tracking_cam_data[2]))
            w = np.array(w)
            error = np.sum((best_cmd_w-w)**2)
            if error < optimal_error_cam:
                optimal_error_cam = error
                optimal_delay_cam = delay
                best_w = w

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

        odom_errors = []
        optimal_delay_odom, optimal_error_odom = 0.0, np.inf
        best_odom_w = []
        for delay in delay_options:
            odom_w = []
            for t in time_points:
                odom_w.append(get_value_at_time(t+delay, odom_data[0], odom_ang_vel))
            odom_w = np.array(odom_w)
            error = np.sum((best_cmd_w-odom_w)**2)
            odom_errors.append(error)
            if (error < optimal_error_odom):
                optimal_error_odom = error
                optimal_delay_odom = delay
                best_odom_w = odom_w
        
    # _, ax = plt.subplots()
    # ax.plot(time_points, best_cmd_w, label="cmd_w")
    # ax.plot(time_points, best_w, label="w")
    # ax.plot(time_points, best_imu_w, label="imu_w")
    # ax.legend()
    
    plt.plot(delay_options, odom_errors)
    plt.show()

    print(subfolder+" imu delay:", optimal_delay_imu)
    print(subfolder+" cam delay:", optimal_delay_cam)
    print(subfolder+" odo delay:", optimal_delay_odom)
    plt.show()  

    return optimal_delay_cam, optimal_delay_imu, optimal_delay_odom


if __name__ == "__main__":
    subfolder = "ikddata2"

    if subfolder == "ikddata2":
        imu_delay = 0.176
        cam_delay = 0.172
        odom_delay = -0.588
    # elif subfolder == "data1":
    #     imu_delay = 0.198
    #     cam_delay = 0.19
    #     odo_delay = -0.324

    # imu_delay, cam_delay, odo_delay = align(subfolder)

    # subfolder = "ikddata2"

    # odom_data = extract_vesc_odom("ikddata2")
    # joystick_data = extract_joystick_data(subfolder)


    # plt.plot(odom_data[0], [i if abs(i) < 20 else 0 for i in odom_data[4]], label="odom")
    # plt.plot(joystick_data[0], joystick_data[2], label="joystick", alpha=0.5)
    # plt.legend()
    # plt.show()

    # visualize_odom(subfolder, cam_delay, odo_delay)
    write_train_data(imu_delay,cam_delay,odom_delay,"ikddata2")   