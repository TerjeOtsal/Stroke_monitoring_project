import numpy as np
import pandas as pd
from scipy.linalg import block_diag
from pykalman import KalmanFilter
import matplotlib.pyplot as plt
import math

# Step 1: Read sensor data from CSV file
df = pd.read_csv('sensor_data_circle.csv')

# Extract necessary columns
timestamps = df['timestamp'].values
acc_data = df[['acc_x', 'acc_y', 'acc_z']].values
gyro_data = df[['gyro_x', 'gyro_y', 'gyro_z']].values
mag_data = df[['mag_x', 'mag_y', 'mag_z']].values  # Magnetometer data

# Calculate time differences between measurements (delta_t)
delta_t = np.diff(timestamps)

# Gravity (assumed constant along the z-axis)
gravity = np.array([0, 0, 9.81])

# Initial values
initial_velocity = np.array([0, 0, 0])  # Start with zero velocity
initial_position = np.array([0, 0, 0])  # Start with unknown position
positions = [initial_position]  # List to store positions

# Function to update position from acceleration
def update_position(acc, prev_vel, prev_pos, dt):
    # Calculate velocity (by integrating acceleration)
    new_vel = prev_vel + acc * dt
    # Calculate position (by integrating velocity)
    new_pos = prev_pos + new_vel * dt
    return new_vel, new_pos

# Function to calculate orientation from gyroscope and magnetometer
def update_orientation(gyro, mag, dt, prev_orientation):
    # Calculate yaw (direction) from the magnetometer
    yaw = math.atan2(mag[1], mag[0])  # Yaw from the magnetometer (just a simple calculation)
    
    # Update pitch and roll from the gyroscope (integration of angular velocity)
    pitch = prev_orientation[0] + gyro[0] * dt
    roll = prev_orientation[1] + gyro[1] * dt
    return np.array([pitch, roll, yaw])

# Function to compensate for gravity based on orientation
def compensate_gravity(acc, orientation):
    pitch, roll, _ = orientation  # Ignore yaw for gravity compensation
    # Rotation matrix to compensate for gravity
    rotation_matrix = np.array([
        [math.cos(pitch), 0, math.sin(pitch)],
        [0, 1, 0],
        [-math.sin(pitch), 0, math.cos(pitch)]
    ])
    
    # Compensate gravity in the body using the rotation matrix
    corrected_acc = np.dot(rotation_matrix, acc - gravity)
    return corrected_acc

# Step 2: Calculate position based on accelerometer and gyroscope data
orientation = np.array([0, 0, 0])  # Start with zero orientation (pitch, roll, yaw)

for i in range(1, len(acc_data)):
    dt = delta_t[i-1]
    
    # Update orientation based on gyroscope and magnetometer
    orientation = update_orientation(gyro_data[i], mag_data[i], dt, orientation)
    
    # Compensate acceleration for gravity
    acc_corrected = compensate_gravity(acc_data[i], orientation)
    
    # Update position based on the corrected acceleration
    velocity, position = update_position(acc_corrected, initial_velocity, positions[-1], dt)
    
    positions.append(position)

positions = np.array(positions)

# Step 3: Kalman filter for smoothing
transition_matrix = np.eye(6)  # State model (position and velocity)
transition_matrix[:3, 3:] = np.eye(3) * np.mean(delta_t)  # Include velocity component

observation_matrix = np.zeros((3, 6))  # Measurement matrix (we observe acceleration)
observation_matrix[:, :3] = np.eye(3)

# Initial states
initial_state_mean = np.hstack([positions[0], np.zeros(3)])  # Use the first observed position and zero velocity

# Kalman filter configuration
kf = KalmanFilter(
    transition_matrices=transition_matrix,
    observation_matrices=observation_matrix,
    initial_state_mean=initial_state_mean,
    observation_covariance=0.1 * np.eye(3),  # Measurement uncertainty
    transition_covariance=0.1 * np.eye(6)    # System uncertainty
)

# Run Kalman filter on position data
filtered_state_means, filtered_state_covariances = kf.filter(positions)

# Extract filtered positions
filtered_positions = filtered_state_means[:, :3]

# Step 4: Save the filtered positions to a new CSV file
filtered_df = pd.DataFrame({
    'timestamp': timestamps,  # Retain original timestamps
    'filtered_x': filtered_positions[:, 0],
    'filtered_y': filtered_positions[:, 1],
    'filtered_z': filtered_positions[:, 2]
})

filtered_df.to_csv('filtered_positions.csv', index=False)
print("Filtered positions are saved in 'filtered_positions.csv'.")

# Step 5: Plot the filtered positions in a 3D plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plot the data in 3D
x = filtered_df['filtered_x']
y = filtered_df['filtered_y']
z = filtered_df['filtered_z']
ax.plot(x, y, z, label='Filtered arm position')

# Labels
ax.set_xlabel('X position (meters)')
ax.set_ylabel('Y position (meters)')
ax.set_zlabel('Z position (meters)')
ax.set_title('Arm movement in 3D over time')

plt.show()
