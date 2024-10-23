import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import cumulative_trapezoid

# CSV-dataen er konvertert til en pandas DataFrame
data = {
    'timestamp': [3.37, 3.93, 4.49, 5.06, 5.62, 6.18, 6.74, 7.32],
    'acc_x': [-8.67, -9.35, -9.56, -9.65, -9.60, -8.97, -8.61, -7.42],
    'acc_y': [3.19, 2.91, 3.94, 1.70, 2.44, 2.71, 4.08, 4.24],
    'acc_z': [1.43, 2.57, 0.22, 0.61, 1.32, 0.98, 2.69, 3.56],
    'gyro_x': [0.04, -0.78, -0.84, -0.33, 0.25, 0.04, 0.50, 0.19],
    'gyro_y': [-0.45, 0.04, -0.20, -0.14, -0.02, 0.11, -0.39, -0.22],
    'gyro_z': [0.23, -0.10, -1.42, 0.11, 0.06, -0.09, 0.88, 0.70]
}

df = pd.DataFrame(data)

# Delta tid, vi bruker np.diff uten prepend og tar gjennomsnitt av nabotider
dt = np.diff(df['timestamp'])
#dt = np.concatenate(([dt[0]], dt))  # Utvid dt til å matche akselerasjonsdataene

# Beregner hastigheten ved å integrere akselerasjonen
vel_x = cumulative_trapezoid(df['acc_x'], dx=dt, initial=0)
vel_y = cumulative_trapezoid(df['acc_y'], dx=dt, initial=0)
vel_z = cumulative_trapezoid(df['acc_z'], dx=dt, initial=0)

# Beregner posisjonen ved å integrere hastigheten
pos_x = cumulative_trapezoid(vel_x, dx=dt, initial=0)
pos_y = cumulative_trapezoid(vel_y, dx=dt, initial=0)
pos_z = cumulative_trapezoid(vel_z, dx=dt, initial=0)

# 3D-plotting
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plott posisjonene
ax.plot(pos_x, pos_y, pos_z, label='Håndleddets bane')
ax.scatter(pos_x, pos_y, pos_z, c='r')  # Marker punktene

# Sett aksetitler
ax.set_xlabel('X-posisjon (m)')
ax.set_ylabel('Y-posisjon (m)')
ax.set_zlabel('Z-posisjon (m)')
ax.set_title('3D-bane for håndleddets bevegelse')

plt.legend()
plt.show()
