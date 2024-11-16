import pandas as pd
import numpy as np

# Load the CSV file
data = pd.read_csv('sensor_data3.csv')

# Convert acceleration from m/s² to g
data['acc_x'] = data['acc_x'] / 9.81
data['acc_y'] = data['acc_y'] / 9.81
data['acc_z'] = data['acc_z'] / 9.81

# Save the converted data to a new CSV file
data.to_csv('converted_sensor_data.csv', index=False)

print("Conversion completed and saved to 'converted_sensor_data.csv'")
