import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt

# Load the data from a CSV file
data = pd.read_csv('sensor_data3.csv').head(10)

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

# Calculate translations relative to the reference point
points = [np.array([0, 0, 0])]  # Start at the reference point

for i in range(1, len(data)):
    acc = data.iloc[i][['acc_x', 'acc_y', 'acc_z']].values
    mag = data.iloc[i][['mag_x', 'mag_y', 'mag_z']].values
    
    # Calculate orientation (not used in plotting but stored for completeness)
    orientation = calculate_orientation(acc, mag)
    orientations.append(orientation)
    
    # Translation vector relative to the reference point
    translation = acc - ref_acc
    translations.append(translation)
    
    # Calculate the position of the current point relative to the reference
    point = np.array([0, 0, 0]) + translation
    points.append(point)

# Convert points to a numpy array for easier plotting
points = np.array(points)

# Plotting
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Plot the points as a line
ax.plot(points[:, 0], points[:, 1], points[:, 2], color='green', marker='o', label='Path')

# Plot the first (reference) point in red
ax.scatter(0, 0, 0, color='red', s=100, label='Reference Point')

# Set plot labels
ax.set_xlabel('X-axis')
ax.set_ylabel('Y-axis')
ax.set_zlabel('Z-axis')
ax.legend()
plt.title("3D Path with Translations Relative to the Reference Point")

# Set axis limits (adjust as needed)
x_limit = (-0.5, 0.5)
y_limit = (-0.5, 0.5)
z_limit = (-0.5, 0.5)

ax.set_xlim(x_limit)
ax.set_ylim(y_limit)
ax.set_zlim(z_limit)

plt.show()
