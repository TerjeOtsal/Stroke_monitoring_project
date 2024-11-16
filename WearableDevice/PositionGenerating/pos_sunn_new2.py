import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt

# Load the data
data = pd.DataFrame({
    'timestamp': [3.49, 3.65, 3.82, 3.98, 4.14, 4.31, 4.47],
    'acc_x': [0.01, 0.15, -0.13, 0.35, 0.28, 0.04, -0.53],
    'acc_y': [1.01, 1.28, 1.09, 1.09, 0.90, 1.02, 0.88],
    'acc_z': [9.84, 9.48, 10.10, 9.91, 9.61, 9.97, 10.06],
    'gyro_x': [0.01, -0.04, -0.02, -0.06, 0.13, -0.13, -0.09],
    'gyro_y': [-0.05, -0.05, -0.03, -0.18, 0.01, 0.09, -0.11],
    'gyro_z': [-0.04, -0.06, -0.11, -0.18, -0.22, -0.55, -0.31],
    'mag_x': [31.98, 32.14, 32.94, 31.99, 32.43, 32.90, 33.21],
    'mag_y': [-27.95, -27.58, -27.71, -27.17, -27.07, -26.19, -24.60],
    'mag_z': [-45.81, -47.81, -46.42, -46.00, -46.99, -46.27, -46.90]
})

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

# Reference point (first row)
ref_acc = data.iloc[0][['acc_x', 'acc_y', 'acc_z']].values
ref_mag = data.iloc[0][['mag_x', 'mag_y', 'mag_z']].values
ref_orientation = calculate_orientation(ref_acc, ref_mag)

# Store translations and orientations
translations = []
orientations = [ref_orientation]

# Iterate over data to calculate orientations and translations
for i in range(1, len(data)):
    acc = data.iloc[i][['acc_x', 'acc_y', 'acc_z']].values
    mag = data.iloc[i][['mag_x', 'mag_y', 'mag_z']].values
    
    orientation = calculate_orientation(acc, mag)
    orientations.append(orientation)
    
    translation = acc - ref_acc
    translations.append(translation)

# Convert to numpy arrays for easier manipulation
translations = np.array(translations)
orientations = np.array(orientations)

# Cumulative sum to get actual points in 3D space
points = np.cumsum(np.vstack(([0, 0, 0], translations)), axis=0)

# Plotting
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Plot the points
ax.plot(points[:, 0], points[:, 1], points[:, 2], color='green', marker='o', label='Path')

# Plot the first (reference) point in red
ax.scatter(0, 0, 0, color='red', s=100, label='Reference Point')

# Set plot labels
ax.set_xlabel('X-axis')
ax.set_ylabel('Y-axis')
ax.set_zlabel('Z-axis')
ax.legend()
plt.title("3D Path from Translations")

# Set axis limits (adjust as needed)
x_limit = (-1, 1)
y_limit = (-1, 1)
z_limit = (-1, 1)

ax.set_xlim(x_limit)
ax.set_ylim(y_limit)
ax.set_zlim(z_limit)

plt.show()
