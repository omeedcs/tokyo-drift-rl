import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
import math

def get_value_at_time(time, times, values):
    if time < times[0]:
        return values[0]
    elif time > times[-1]:
        return values[-1]
    left, right = 0, len(times)-1
    while right-left > 1:
        mid = (left+right)//2
        if time > times[mid]:
            left = mid
        elif time < times[mid]:
            right = mid
        else:
            return values[mid]
    return values[left]+(values[right]-values[left])*(time-times[left])/(times[right]-times[left])

def euler_from_quaternion(x, y, z, w):
    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    roll_x = np.arctan2(t0, t1)
    
    # t2 = +2.0 * (w * y - z * x)
    # t2 
    # t2 = +1.0 if t2 > +1.0 else t2
    # t2 = -1.0 if t2 < -1.0 else t2
    # pitch_y = np.arcsin(t2)
    
    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    yaw_z = np.arctan2(t3, t4)
    
    return roll_x, 0, yaw_z # in radians

def subt_poses(x, y, t, x2, y2, t2):
    x2 -= x
    y2 -= y

    x2 = x2*np.cos(-t) - y2*np.sin(-t)
    y2 = y2*np.cos(-t) + x2*np.sin(-t)

    t2 -= t

    return x2, y2, t2