import pandas as pd
import numpy as np
import joblib
import tensorflow as tf
import matplotlib.pyplot as plt

# Step 1: Load the saved model and scaler
model = tf.keras.models.load_model('complex_stroke_movement_classifier.h5')
scaler = joblib.load('scaler.joblib')

# Step 2: Load and preprocess the activity data
# Load the new activity data (make sure columns match the training data except for labels)
activity_data = pd.read_csv('ACTIVITY2-2_no_mag.csv')

# Scale the activity data using the loaded scaler to ensure consistency with training data
activity_data_scaled = scaler.transform(activity_data)

# Step 3: Reshape data for the model (samples, timesteps, features)
# Ensure the timesteps match the setting used during training (e.g., 10 or 20)
timesteps = 5
activity_data_reshaped = np.array([activity_data_scaled[i:i + timesteps] 
                                   for i in range(len(activity_data_scaled) - timesteps)])

# Step 4: Predict movement classes for each sequence
predictions = np.argmax(model.predict(activity_data_reshaped), axis=1)

# Step 5: Map class numbers to descriptive labels
class_names = ['Hand towards body', 'Hand down', 'Hand outwards', 'Hand upwards', 'Hand forward']
predicted_labels = [class_names[pred] for pred in predictions]

# Step 6: Plot the predicted movement class over time
plt.figure(figsize=(12, 6))
plt.plot(range(len(predicted_labels)), predictions, marker='o', linestyle='-', color='b')
plt.yticks(ticks=range(5), labels=class_names)
plt.xlabel('Time Step (sequence)')
plt.ylabel('Predicted Movement Class')
plt.title('Predicted Movement Class Over Time')
plt.grid(True)
plt.show()

# Optional: Display predictions as a DataFrame for inspection
predictions_df = pd.DataFrame({'Time Step': range(len(predicted_labels)), 
                               'Predicted Class': predictions, 
                               'Predicted Label': predicted_labels})
print(predictions_df.head())