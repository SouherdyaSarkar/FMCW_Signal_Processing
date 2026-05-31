import os
import sys
sys.path.append(os.path.join(os.getcwd(), '../..'))
import pandas as pd
import numpy as np
from scipy.integrate import cumulative_trapezoid
import matplotlib.pyplot as plt
from ahrs.filters import Madgwick
from ahrs import QuaternionArray

DATA_PATH="data/IMU_data/straight_Line_shape.csv"
FILENAME=DATA_PATH.split("/")[-1].split(".")[0]

df = pd.read_csv(DATA_PATH)

t = df['timestamp'].values
a_x, a_y, a_z = df['accel_x'].values, df['accel_y'].values, df['accel_z'].values
g_x, g_y, g_z = df['gyro_x'].values, df['gyro_y'].values, df['gyro_z'].values

# dt = np.mean(np.diff(t))
# g_x = g_x - g_x[0]
# g_y = g_y - g_y[0]
# g_z = g_z -  g_z[0]

# gyro_data = np.column_stack((g_x, g_y, g_z))
# accel_data = np.column_stack((a_x, a_y, a_z))

# madgwick = Madgwick(gyr=gyro_data, acc=accel_data, Dt=dt, gain=0.033)
# Q = madgwick.Q 
# R = QuaternionArray(Q).to_DCM()

# global_accel = np.zeros_like(accel_data)
# for i in range(len(t)):
#     global_accel[i] = R[i] @ accel_data[i]

# global_accel[:, 2] -= 9.81

initial_roll = np.arctan2(a_y[0], a_z[0])
initial_pitch = np.arctan2(-a_x[0], np.sqrt(a_y[0]**2 + a_z[0]**2))
initial_yaw = 0.0 
roll_changes = cumulative_trapezoid(g_x, x=t, initial=0)
pitch_changes = cumulative_trapezoid(g_y, x=t, initial=0)
yaw_changes = cumulative_trapezoid(g_z, x=t, initial=0)

roll = initial_roll + roll_changes
pitch = initial_pitch + pitch_changes
yaw = initial_yaw + yaw_changes
true_a_x = a_x - (9.81 * np.sin(pitch))
true_a_y = a_y - (9.81 * np.sin(roll))
true_a_z = a_z - (9.81 * np.cos(pitch) * np.cos(roll))

global_a_x = (true_a_x * np.cos(yaw)) - (true_a_y * np.sin(yaw))
global_a_y = (true_a_x * np.sin(yaw)) + (true_a_y * np.cos(yaw))
global_a_z = true_a_z
v_x = cumulative_trapezoid(global_a_x, x=t, initial=0)
v_y = cumulative_trapezoid(global_a_y, x=t, initial=0)
v_z = cumulative_trapezoid(global_a_z, x=t, initial=0)


p_x = cumulative_trapezoid(v_x, x=t, initial=0)
p_y = cumulative_trapezoid(v_y, x=t, initial=0)
p_z = cumulative_trapezoid(v_z, x=t, initial=0)

SAVE_PATH=f"Simulations/IMU_trajectories/{FILENAME}.png"

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

ax.plot(p_x, p_y, p_z, label='Trajectory', color='blue', linewidth=2)

ax.scatter(p_x[0], p_y[0], p_z[0], color='green', s=50, label='Start')
ax.scatter(p_x[-1], p_y[-1], p_z[-1], color='red', s=50, label='End')

ax.set_xlabel('Position X (mm)')
ax.set_ylabel('Position Y (mm)')
ax.set_zlabel('Position Z (mm)')
ax.set_title(f'3D Trajectory {FILENAME}')
ax.legend()

fig.savefig(SAVE_PATH)

plt.show()

