import numpy as np
import pandas as pd
from scipy.linalg import block_diag
from pykalman import KalmanFilter
import matplotlib.pyplot as plt
import math

# Steg 1: Les sensordata fra CSV-fil
df = pd.read_csv('sensor_data3.csv')

# Ekstraher nødvendige kolonner
timestamps = df['timestamp'].values
acc_data1 = df[['acc_x', 'acc_y', 'acc_z']].values
gyro_data1 = df[['gyro_x', 'gyro_y', 'gyro_z']].values
mag_data1 = df[['mag_x', 'mag_y', 'mag_z']].values  # Magnetometer data

# Normalize data using z-score normalization
def normalize_zscore(data):
    return (data - data.mean()) / data.std()

#  Apply normalization to each dataset (no .values needed here)
acc_data = normalize_zscore(acc_data1)
gyro_data = normalize_zscore(gyro_data1)
mag_data = normalize_zscore(mag_data1)

# Beregn tidsdifferanser mellom målinger (delta_t)
delta_t = np.diff(timestamps)

# Gravitasjon (antatt konstant langs z-aksen)
gravity = np.array([0, 0, 9.81])

# Startverdier
initial_velocity = np.array([0, 0, 0])  # Start med null hastighet
initial_position = np.array([0, 0, 0])  # Start med ukjent posisjon
positions = [initial_position]  # Liste for å lagre posisjonene

# Funksjon for å oppdatere posisjon fra akselerasjon
def update_position(acc, start_pos, dt, index):
    """
    Beregn posisjon i forhold til startposisjonen (første punkt) hver gang.
    :param acc: Akselerasjonen ved nåværende tidspunkt
    :param start_pos: Startposisjonen, dvs. første punkt
    :param dt: Tidsdifferansen til dette tidspunktet
    :param index: Indeksen for nåværende tidspunkt
    :return: Beregnet posisjon relativt til startposisjon
    """
    # Akkumuler posisjon relativt til startpunkt ved integrering av akselerasjonen over tid
    # Vi antar at start velocity er null
    velocity = acc * dt * index  # Enkel tilnærming der hastighet akkumuleres over tid
    position = start_pos + velocity * dt * index  # Posisjonen relatert til startpunkt
    return position

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
    
    # Oppdater posisjonen basert på den korrigerte akselerasjonen relativt til startposisjonen
    position = update_position(acc_corrected, initial_position, dt, i)
    
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
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plot startposisjonen som et større punkt
start_x = filtered_df['filtered_x'].iloc[0]
start_y = filtered_df['filtered_y'].iloc[0]
start_z = filtered_df['filtered_z'].iloc[0]
ax.scatter(start_x, start_y, start_z, color='red', s=50, label='Startposisjon')  # Større og annen farge for startpunktet

# Plot de resterende punktene i 3D uten linjer
x = filtered_df['filtered_x'][1:]  # Alle unntatt startposisjon
y = filtered_df['filtered_y'][1:]
z = filtered_df['filtered_z'][1:]
ax.scatter(x, y, z, color='blue', s=5, label='Filtrert armposisjon')  # 's' er størrelsen på punktene

# Etiketter
ax.set_xlabel('X posisjon (meter)')
ax.set_ylabel('Y posisjon (meter)')
ax.set_zlabel('Z posisjon (meter)')
ax.set_title('Armens bevegelse i 3D over tid')
ax.legend()

plt.show()