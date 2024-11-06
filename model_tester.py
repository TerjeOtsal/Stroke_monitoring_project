import pandas as pd
import numpy as np
import joblib
import tensorflow as tf
import matplotlib.pyplot as plt

# Load the saved model and scaler
model = tf.keras.models.load_model('complex_stroke_movement_classifier.h5')
scaler = joblib.load('scaler.joblib')

# Load and preprocess the activity data
activity_data = pd.read_csv('ACTIVITY.csv')
activity_data_scaled = scaler.transform(activity_data)

# Define the same number of timesteps used during training
timesteps = 10

# Reshape data for the model (samples, timesteps, features)
activity_data_reshaped = np.array([activity_data_scaled[i:i + timesteps] 
                                   for i in range(len(activity_data_scaled) - timesteps)])

# Predict movement classes for each sequence
predictions = np.argmax(model.predict(activity_data_reshaped), axis=1)

# Map class numbers to labels
class_names = ['Hand towards body', 'Hand down', 'Hand outwards', 'Hand upwards', 'Hand forward']
predicted_labels = [class_names[pred] for pred in predictions]

# Plot the predicted movement class over time
plt.figure(figsize=(12, 6))
plt.plot(range(len(predicted_labels)), predictions, marker='o', linestyle='-', color='b')
plt.yticks(ticks=range(5), labels=class_names)
plt.xlabel('Time Step (sequence)')
plt.ylabel('Predicted Movement Class')
plt.title('Predicted Movement Class Over Time')
plt.grid(True)
plt.show()
