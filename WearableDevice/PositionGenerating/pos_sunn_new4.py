import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load the data from a CSV file and only take the first 10 points
data = pd.read_csv('sensor_data3.csv')

# Function to calculate roll, pitch, and yaw from accelerometer and magnetometer data
def calculate_orientation(acc, mag):
    acc = acc / np.linalg.norm(acc)
    mag = mag / np.linalg.norm(mag)
    
    pitch = np.arcsin(-acc[0])
    roll = np.arctan2(acc[1], acc[2])
    
    mag_x = mag[0] * np.cos(pitch) + mag[2] * np.sin(pitch)
    mag_y = mag[0] * np.sin(roll) * np.sin(pitch) + mag[1] * np.cos(roll) - mag[2] * np.sin(roll) * np.cos(pitch)
    yaw = np.arctan2(-mag_y, mag_x)
    
    return roll, pitch, yaw

# Simple Kalman filter implementation
def kalman_filter(data, process_var=1e-5, measurement_var=1e-1):
    n = len(data)
    estimated = np.zeros(n)
    est_var = np.zeros(n)
    
    # Initial estimates
    estimated[0] = data[0]
    est_var[0] = 1.0
    
    for i in range(1, n):
        # Prediction step
        estimated[i] = estimated[i - 1]  # Predicted state is the previous state
        est_var[i] = est_var[i - 1] + process_var  # Predicted variance
        
        # Update step
        kalman_gain = est_var[i] / (est_var[i] + measurement_var)
        estimated[i] = estimated[i] + kalman_gain * (data[i] - estimated[i])
        est_var[i] = (1 - kalman_gain) * est_var[i]
        
    return estimated

# Reference point (first row)
ref_acc = data.iloc[0][['acc_x', 'acc_y', 'acc_z']].values
ref_mag = data.iloc[0][['mag_x', 'mag_y', 'mag_z']].values
ref_orientation = calculate_orientation(ref_acc, ref_mag)

# Store translations relative to the reference point
points = [np.array([0, 0, 0])]  # Start at the reference point

for i in range(1, len(data)):
    acc = data.iloc[i][['acc_x', 'acc_y', 'acc_z']].values
    
    # Translation vector relative to the reference point (always using the first row)
    translation = acc - ref_acc
    
    # Add translation as a point relative to the reference
    points.append(translation)

# Apply Kalman filter to each axis of the translation points
points = np.array(points)
smoothed_points = np.zeros_like(points)

for axis in range(points.shape[1]):
    smoothed_points[:, axis] = kalman_filter(points[:, axis])

# Save the smoothed points to a CSV file
smoothed_points_df = pd.DataFrame(smoothed_points, columns=['x', 'y', 'z'])
smoothed_points_df.to_csv('smoothed_points.csv', index=False)

# Plotting
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Plot the smoothed points as a line
ax.plot(smoothed_points[:, 0], smoothed_points[:, 1], smoothed_points[:, 2], color='blue', marker='o', label='Smoothed Path')

# Plot the first (reference) point in red
ax.scatter(0, 0, 0, color='red', s=100, label='Reference Point')

# Set plot labels
ax.set_xlabel('X-axis')
ax.set_ylabel('Y-axis')
ax.set_zlabel('Z-axis')
ax.legend()
plt.title("3D Path with Kalman Filter Applied (Relative to Reference Point)")

# Set axis limits (adjust as needed)
x_limit = (-1, 1)
y_limit = (-1, 1)
z_limit = (-1, 1)

ax.set_xlim(x_limit)
ax.set_ylim(y_limit)
ax.set_zlim(z_limit)

plt.show()
