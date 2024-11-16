import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt

# Load the data from a CSV file and only take the first 10 points
data = pd.read_csv('sensor_data3.csv')


# Function to calculate roll, pitch, and yaw from accelerometer and magnetometer data
def calculate_orientation(acc, mag):
    # Normalize accelerometer and magnetometer data
    acc = acc / np.linalg.norm(acc)
    mag = mag / np.linalg.norm(mag)
    
    # Calculate pitch and roll from accelerometer
    pitch = np.arcsin(-acc[0])
    roll = np.arctan2(acc[1], acc[2])
    
    # Calculate yaw using magnetometer data
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
    
    # Calculate orientation (roll, pitch, yaw)
    orientation = calculate_orientation(acc, mag)
    orientations.append(orientation)
    
    # Translation vector from the reference point
    translation = acc - ref_acc
    translations.append(translation)

# Convert to numpy arrays for easier manipulation
translations = np.array(translations)
orientations = np.array(orientations)

# Plotting
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Plot the first (reference) point in red
ax.scatter(0, 0, 0, color='red', s=100, label='Reference Point')

# Plot translation vectors for all points (start from reference)
for i in range(len(translations)):
    translation = translations[i]
    ax.quiver(0, 0, 0, translation[0], translation[1], translation[2], color='blue', alpha=0.6)

# Set plot labels
ax.set_xlabel('X-axis')
ax.set_ylabel('Y-axis')
ax.set_zlabel('Z-axis')
ax.legend()
plt.title("3D Translation Vectors from Reference Point")

# Set axis limits (adjust as needed)
x_limit = (-5, 5)
y_limit = (-2, 1)
z_limit = (-2, 1)

ax.set_xlim(x_limit)
ax.set_ylim(y_limit)
ax.set_zlim(z_limit)

plt.show()
