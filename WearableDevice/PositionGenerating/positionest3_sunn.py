import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Load sensor data from CSV file
df = pd.read_csv('sensor_data3.csv')

# Extract necessary columns
timestamps = df['timestamp'].values
acc_data = df[['acc_x', 'acc_y', 'acc_z']].values
gyro_data = df[['gyro_x', 'gyro_y', 'gyro_z']].values
mag_data = df[['mag_x', 'mag_y', 'mag_z']].values

# Constants
dt = np.mean(np.diff(timestamps))  # Sample time interval

# Initialize arrays for roll, pitch, yaw
roll = np.zeros(len(timestamps))
pitch = np.zeros(len(timestamps))
yaw = np.zeros(len(timestamps))

# Complementary Filter coefficients
alpha = 0.98

# Step 1: Calculate roll and pitch using accelerometer data
for i in range(1, len(timestamps)):
    acc_x, acc_y, acc_z = acc_data[i]
    
    # Using the second algorithm to compute roll and pitch
    roll[i] = np.arctan2(acc_y, acc_z)
    pitch[i] = np.arctan2(-acc_x, np.sqrt(acc_y**2 + acc_z**2))

# Step 2: Calculate roll and pitch using gyroscope data
gyr_roll = np.zeros(len(timestamps))
gyr_pitch = np.zeros(len(timestamps))

for i in range(1, len(timestamps)):
    gyro_x, gyro_y, gyro_z = gyro_data[i]
    
    # Integrate gyroscope data to get angles
    gyr_roll[i] = gyr_roll[i-1] + gyro_y * dt
    gyr_pitch[i] = gyr_pitch[i-1] + gyro_x * dt

# Step 3: Apply complementary filter
filtered_roll = alpha * gyr_roll + (1 - alpha) * roll
filtered_pitch = alpha * gyr_pitch + (1 - alpha) * pitch

# Step 4: Calculate yaw using magnetometer data
for i in range(1, len(timestamps)):
    mag_x, mag_y, mag_z = mag_data[i]
    
    # Calculate yaw
    XH = mag_x * np.cos(filtered_pitch[i]) + mag_y * np.sin(filtered_pitch[i]) * np.sin(filtered_roll[i]) + mag_z * np.sin(filtered_pitch[i]) * np.cos(filtered_roll[i])
    YH = mag_y * np.cos(filtered_roll[i]) + mag_z * np.sin(filtered_roll[i])
    yaw[i] = np.arctan2(-YH, XH)

# Step 5: Calculate linear acceleration
linear_acc = np.zeros_like(acc_data)
for i in range(len(timestamps)):
    # Compensate for gravity
    g = 9.81  # gravity in m/s^2
    acc_x = acc_data[i, 0] - g * np.sin(pitch[i])  # compensate for gravity
    acc_y = acc_data[i, 1] + g * np.cos(pitch[i]) * np.sin(roll[i])
    acc_z = acc_data[i, 2] + g * np.cos(pitch[i]) * np.cos(roll[i])
    
    linear_acc[i] = [acc_x, acc_y, acc_z]

# Step 6: Integrate acceleration to get velocity
velocity = np.zeros_like(linear_acc)
for i in range(1, len(timestamps)):
    velocity[i] = velocity[i-1] + linear_acc[i] * dt

# Step 7: Integrate velocity to get position
position = np.zeros_like(velocity)
for i in range(1, len(timestamps)):
    position[i] = position[i-1] + velocity[i] * dt

# Step 8: Normalize the position data using Min-Max normalization
min_vals = np.min(position, axis=0)
max_vals = np.max(position, axis=0)
normalized_position = (position - min_vals) / (max_vals - min_vals)

# Step 9: Plot the normalized position in 3D
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.plot(normalized_position[:, 0], normalized_position[:, 1], normalized_position[:, 2], label='3D Trajectory', color='b')
ax.set_xlabel('Normalized X Position')
ax.set_ylabel('Normalized Y Position')
ax.set_zlabel('Normalized Z Position')
ax.set_title('Normalized 3D Position Plot')
ax.legend()
plt.show()
