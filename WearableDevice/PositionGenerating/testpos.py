import numpy as np
import pandas as pd
from scipy.linalg import block_diag
from pykalman import KalmanFilter
import matplotlib.pyplot as plt

# Steg 1: Les sensordata fra CSV-fil
df = pd.read_csv('sensor_data.csv')

# Vis kolonnene
print(df.head())

# Ekstraher nødvendige kolonner
timestamps = df['timestamp'].values
acc_data = df[['acc_x', 'acc_y', 'acc_z']].values
gyro_data = df[['gyro_x', 'gyro_y', 'gyro_z']].values