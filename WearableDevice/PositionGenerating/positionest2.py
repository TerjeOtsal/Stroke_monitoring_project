import numpy as np
import pandas as pd
from scipy.linalg import block_diag
from pykalman import KalmanFilter
import matplotlib.pyplot as plt

# Steg 1: Les sensordata fra CSV-fil
df = pd.read_csv('sensor_data2.csv')

# Ekstraher nødvendige kolonner
timestamps = df['timestamp'].values
acc_data = df[['acc_x', 'acc_y', 'acc_z']].values
gyro_data = df[['gyro_x', 'gyro_y', 'gyro_z']].values

# Beregn tidsdifferanser mellom målinger (delta_t)
#delta_t = np.diff(timestamps)  # Antatt at tidsstempel allerede er i sekunder
delta_t = np.diff(timestamps) / 1000.0 

# Funksjon for å oppdatere posisjon fra akselerasjon
def update_position(acc, prev_vel, prev_pos, dt):
    # Beregn hastighet (ved å integrere akselerasjon)
    new_vel = prev_vel + acc * dt
    # Beregn posisjon (ved å integrere hastighet)
    new_pos = prev_pos + new_vel * dt
    return new_vel, new_pos

# Gravitasjon (kan justeres basert på måleforholdene)
gravity = np.array([0, 0, 9.81])  # Standard tyngdekraft i z-retning (kan være nødvendigvis i lokale akser)

# Start med null hastighet og posisjon
initial_velocity = np.array([0, 0, 0])  # Start med null hastighet
initial_position = np.array([0, 0, 0])  # Start med ukjent posisjon
positions = [initial_position]  # Liste for å lagre posisjonene

# Steg 2: Beregn posisjon basert på akselerometerdataene
for i in range(1, len(acc_data)):
    acc = acc_data[i] - gravity  # Fjern gravitasjonen fra akselerasjonen
    dt = delta_t[i-1]
    velocity, position = update_position(acc, initial_velocity, positions[-1], dt)
    
    # Begrens posisjon innenfor armens rekkevidde (55 cm)
    #if np.linalg.norm(position) <= 0.55:  # 55 cm rekkevidde
    #    positions.append(position)
    #else:
    #    positions.append(positions[-1])  # Behold forrige posisjon dersom vi er utenfor rekkevidden

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
    observation_covariance=0.1 * np.eye(3),  # Måling usikkerhet
    transition_covariance=0.1 * np.eye(6)  # Systemets usikkerhet
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

# Lagre som ny csv-fil
filtered_df.to_csv('filtered_positions.csv', index=False)

print("Filtrerte posisjoner er lagret i 'filtered_positions.csv'.")

# Steg 5: Plot de filtrerte posisjonene i et 3D-plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plot dataene i 3D
x = filtered_df['filtered_x']
y = filtered_df['filtered_y']
z = filtered_df['filtered_z']
ax.plot(x, y, z, label='Filtrert armposisjon')

# Etiketter
ax.set_xlabel('X posisjon (meter)')
ax.set_ylabel('Y posisjon (meter)')
ax.set_zlabel('Z posisjon (meter)')
ax.set_title('Armens bevegelse i 3D over tid')

plt.show()
