import pandas as pd
import numpy as np
import tensorflow as tf
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix

# Load the saved model and scaler
model = tf.keras.models.load_model('cnn_stroke_movement_classifier.keras')
scaler = joblib.load('scaler.joblib')

# Define the number of timesteps used during training
timesteps = 10

# Step 1: Load and Preprocess the Activity Data
# Load the Activity.csv file (unlabeled)
activity_data = pd.read_csv('Activity.csv')  # Replace with the actual path if needed

# Scale the activity data using the loaded scaler
activity_data_scaled = scaler.transform(activity_data)

# Reshape data to match the model’s expected input shape (samples, timesteps, features)
activity_data_reshaped = np.array([activity_data_scaled[i:i + timesteps] for i in range(len(activity_data_scaled) - timesteps)])

# Step 2: Make Predictions
# Get model predictions for each time window in the Activity data
predictions = np.argmax(model.predict(activity_data_reshaped), axis=1)

# Map class numbers to labels for interpretation
class_names = ['Hand towards body', 'Hand down', 'Hand outwards', 'Hand upwards', 'Hand forward']
predicted_labels = [class_names[pred] for pred in predictions]

# Step 3: Plot Predicted Classes Over Time
plt.figure(figsize=(12, 6))
plt.plot(range(len(predicted_labels)), predictions, marker='o', linestyle='-', color='b')
plt.yticks(ticks=range(5), labels=class_names)
plt.xlabel('Time Step (sequence)')
plt.ylabel('Predicted Movement Class')
plt.title('Predicted Movement Class Over Time for Activity Data')
plt.grid(True)
plt.tight_layout()
plt.show()

# Optional: Display predictions as a DataFrame for inspection
predictions_df = pd.DataFrame({
    'Time Step': range(len(predicted_labels)), 
    'Predicted Class': predictions, 
    'Predicted Label': predicted_labels
})
print(predictions_df.head())  # Display the first few predictions
