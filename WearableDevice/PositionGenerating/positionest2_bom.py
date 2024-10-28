import numpy as np
import pandas as pd
from pykalman import KalmanFilter
import math
import matplotlib.pyplot as plt

# Steg 1: Les sensordata fra CSV-fil
df = pd.read_csv('sensor_data3.csv')

# Ekstraher nødvendige kolonner
timestamps = df['timestamp'].values
acc_data = df[['acc_x', 'acc_y', 'acc_z']].values
gyro_data = df[['gyro_x', 'gyro_y', 'gyro_z']].values
mag_data = df[['mag_x', 'mag_y', 'mag_z']].values  # Magnetometer data

# Beregn tidsdifferanser mellom målinger (delta_t)
delta_t = np.diff(timestamps)

def normalize(data):
    return (data - np.min(data, axis=0)) / (np.max(data, axis=0) - np.min(data, axis=0))

# Normalize accelerometer and gyroscope data
acc_data = normalize(acc_data)
gyro_data = normalize(gyro_data)
# Gravitasjon (antatt konstant langs z-aksen)
gravity = np.array([0, 0, 9.81])

# Startverdier
initial_velocity = np.array([0, 0, 0])  # Start med null hastighet
initial_position = np.array([0, 0, 0])  # Start med ukjent posisjon
positions = [initial_position]  # Liste for å lagre posisjonene

# Funksjon for å oppdatere posisjon fra akselerasjon
def update_position(acc, prev_vel, prev_pos, dt):
    # Beregn hastighet (ved å integrere akselerasjon)
    new_vel = prev_vel + acc * dt
    # Beregn posisjon (ved å integrere hastighet)
    new_pos = prev_pos + new_vel * dt
    return new_vel, new_pos

# Funksjon for å beregne orientering fra gyroskop og magnetometer
def update_orientation(gyro, mag, dt, prev_orientation):
    # Beregn yaw (retningen) fra magnetometeret
    yaw = math.atan2(mag[1], mag[0])  # Yaw fra magnetometeret (bare en enkel beregning)
    
    # Oppdater pitch og roll fra gyroskop (integrasjon av vinkelhastighet)
    pitch = prev_orientation[0] + gyro[0] * dt
    roll = prev_orientation[1] + gyro[1] * dt
    return np.array([pitch, roll, yaw])

# Funksjon for å kompensere for gravitasjon basert på orientering
def compensate_gravity(acc, orientation):
    pitch, roll, _ = orientation  # Vi ignorerer yaw for gravitasjon
    # Rotasjonsmatrise for å kompensere gravitasjon
    rotation_matrix = np.array([
        [math.cos(pitch), 0, math.sin(pitch)],
        [0, 1, 0],
        [-math.sin(pitch), 0, math.cos(pitch)]
    ])
    
    # Kompensere gravitasjon i kroppen ved å bruke rotasjonsmatrisen
    corrected_acc = np.dot(rotation_matrix, acc - gravity)
    return corrected_acc

# Steg 2: Beregn posisjon basert på akselerometer- og gyrodata
orientation = np.array([0, 0, 0])  # Start med null orientering (pitch, roll, yaw)

for i in range(1, len(acc_data)):
    dt = delta_t[i-1]
    
    # Oppdater orienteringen basert på gyroskop og magnetometer
    orientation = update_orientation(gyro_data[i], mag_data[i], dt, orientation)
    
    # Kompenser akselerasjon for gravitasjon
    acc_corrected = compensate_gravity(acc_data[i], orientation)
    
    # Oppdater posisjonen basert på den korrigerte akselerasjonen
    velocity, position = update_position(acc_corrected, initial_velocity, positions[-1], dt)
    
    positions.append(position)

positions = np.array(positions)

# Steg 3: Kalman-filter for glatting
transition_matrix = np.eye(6)  # Tilstandsmodellen (posisjon og hastighet)
transition_matrix[:3, 3:] = np.eye(3) * np.mean(delta_t)  # Inkluder hastighetskomponenten

observation_matrix = np.zeros((3, 6))  # Målematrisen (vi observerer akselerasjon)
observation_matrix[:, :3] = np.eye(3)

# Starttilstander
initial_state_mean = np.hstack([positions[0], np.zeros(3)])  # Bruk første observerte posisjon og null-hastighet

# Kalman-filter konfigurering
kf = KalmanFilter(
    transition_matrices=transition_matrix,
    observation_matrices=observation_matrix,
    initial_state_mean=initial_state_mean,
    observation_covariance=0.1 * np.eye(3),  # Måleusikkerhet
    transition_covariance=0.1 * np.eye(6)    # Systemets usikkerhet
)

# Kjør Kalman-filter på posisjonsdata
filtered_state_means, filtered_state_covariances = kf.filter(positions)

# Ekstraher filtrerte posisjoner
filtered_positions = filtered_state_means[:, :3]

# Steg 4: Lagre de filtrerte posisjonene i en ny CSV-fil
filtered_df = pd.DataFrame({
    'timestamp': timestamps,  # Behold originale tidsstempler
    'filtered_x': filtered_positions[:, 0],
    'filtered_y': filtered_positions[:, 1],
    'filtered_z': filtered_positions[:, 2]
})

filtered_df.to_csv('filtered_positions.csv', index=False)
print("Filtrerte posisjoner er lagret i 'filtered_positions.csv'.")

# Steg 5: Plot de filtrerte posisjonene i et 3D-plot
import matplotlib.pyplot as plt

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plot dataene som punkter i 3D
x = filtered_df['filtered_x']
y = filtered_df['filtered_y']
z = filtered_df['filtered_z']

# Increase the size of the points for better visibility
point_size = 5  # Adjust the size as needed
ax.scatter(x, y, z, label='Filtrert armposisjon', color='b', s=point_size)  # Use scatter for individual points

# Etiketter
ax.set_xlabel('X posisjon (meter)')
ax.set_ylabel('Y posisjon (meter)')
ax.set_zlabel('Z posisjon (meter)')
ax.set_title('Filtrerte posisjoner i 3D')

# Set the limits for each axis to zoom in on the data
ax.set_xlim([min(x) - 0.5, max(x) + 0.5])  # Adjust limits to zoom in on differences
ax.set_ylim([min(y) - 0.5, max(y) + 0.5])  # Adjust limits to zoom in on differences
ax.set_zlim([min(z) - 0.5, max(z) + 0.5])  # Adjust limits to zoom in on differences

# Optional: Adjust the viewing angle for better perspective
ax.view_init(elev=100, azim=20)  # Elevation and azimuthal angle

plt.legend()
plt.show()
