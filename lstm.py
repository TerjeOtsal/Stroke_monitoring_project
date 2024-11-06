import pandas as pd
import numpy as np
import joblib
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt

# Load and preprocess labeled data
data = pd.read_csv('labeled_stroke_data.csv')
X = data.drop(columns=['label'])
y = data['label'] - 1  # Adjusting label range to start from 0 for TensorFlow

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Reshape data for CNN-LSTM model (samples, timesteps, features)
timesteps = 10  # Define timesteps for LSTM sequence processing
X_reshaped = np.array([X_scaled[i:i + timesteps] for i in range(len(X_scaled) - timesteps)])
y_reshaped = y[timesteps:].values

# Split data
X_train, X_val, y_train, y_val = train_test_split(X_reshaped, y_reshaped, test_size=0.2, random_state=42)

# Define CNN-LSTM model architecture
model = tf.keras.Sequential([
    tf.keras.layers.Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(timesteps, X_reshaped.shape[2])),
    tf.keras.layers.MaxPooling1D(pool_size=2),
    tf.keras.layers.Conv1D(filters=32, kernel_size=3, activation='relu'),
    tf.keras.layers.MaxPooling1D(pool_size=2),
    tf.keras.layers.LSTM(50, activation='relu', return_sequences=True),
    tf.keras.layers.LSTM(50, activation='relu'),
    tf.keras.layers.Dense(50, activation='relu'),
    tf.keras.layers.Dense(5, activation='softmax')  # 5 classes for classification
])

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(X_train, y_train, epochs=30, validation_data=(X_val, y_val), batch_size=32, verbose=1)

# Evaluate the model
val_loss, val_accuracy = model.evaluate(X_val, y_val)
y_pred = np.argmax(model.predict(X_val), axis=1)

# Classification report and confusion matrix
print("Classification Report:\n", classification_report(y_val, y_pred, target_names=[
    'Hand towards body', 'Hand down', 'Hand outwards', 'Hand upwards', 'Hand forward']))
print("Confusion Matrix:\n", confusion_matrix(y_val, y_pred))

# Save the model and scaler
model.save('complex_stroke_movement_classifier.h5')
joblib.dump(scaler, 'scaler.joblib')

# Plotting training and validation accuracy over epochs
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Training and Validation Accuracy Over Epochs')
plt.show()

# Plotting training and validation loss over epochs
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Training and Validation Loss Over Epochs')
plt.show()
