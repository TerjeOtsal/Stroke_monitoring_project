import pandas as pd
import numpy as np
import joblib
import tensorflow as tf
from tensorflow import keras 
from keras import regularizers
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt

# Step 1: Load and Preprocess Data
# Load the extended dataset with integer labels
data = pd.read_csv('extended_labeled_stroke_data_int_labels.csv')

# Separate features (X) and labels (y)
X = data.drop(columns=['label'])
y = data['label'] - 1  # Adjust labels to start from 0 for TensorFlow compatibility

# Step 2: Standardize Features
# Initialize the scaler and fit-transform the features for normalization
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 3: Reshape Data for CNN-LSTM Model
# Define the number of timesteps for LSTM sequence processing
timesteps = 10

# Reshape data into a 3D array: (samples, timesteps, features)
X_reshaped = np.array([X_scaled[i:i + timesteps] for i in range(len(X_scaled) - timesteps)])
y_reshaped = y[timesteps:].values  # Adjust y to align with reshaped X

# Step 4: Split Data into Training and Validation Sets
# Use an 80-20 split for training and validation
X_train, X_val, y_train, y_val = train_test_split(X_reshaped, y_reshaped, test_size=0.2, random_state=42)

# Step 5: Define CNN-LSTM Model Architecture with Enhanced Robustness
model = tf.keras.Sequential([
    # First Conv1D layer with Batch Normalization and Dropout
    tf.keras.layers.Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(timesteps, X_reshaped.shape[2])),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling1D(pool_size=2),
    tf.keras.layers.Dropout(0.3),
    
    # Second Conv1D layer with Batch Normalization and Dropout
    tf.keras.layers.Conv1D(filters=32, kernel_size=3, activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling1D(pool_size=2),
    tf.keras.layers.Dropout(0.3),
    
    # First LSTM layer with L2 regularization and Batch Normalization
    tf.keras.layers.LSTM(50, activation='relu', return_sequences=True, kernel_regularizer=regularizers.l2(0.01)),
    tf.keras.layers.BatchNormalization(),
    
    # Second LSTM layer with L2 regularization and Batch Normalization
    tf.keras.layers.LSTM(50, activation='relu', kernel_regularizer=regularizers.l2(0.01)),
    tf.keras.layers.BatchNormalization(),
    
    # Dense layer with L2 regularization and Dropout
    tf.keras.layers.Dense(50, activation='relu', kernel_regularizer=regularizers.l2(0.01)),
    tf.keras.layers.Dropout(0.3),
    
    # Output layer for 5 classes with softmax activation
    tf.keras.layers.Dense(5, activation='softmax')
])

# Step 6: Compile the Model
# Use Adam optimizer and sparse categorical crossentropy loss for multi-class classification
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Step 7: Train the Model
# Train the model with specified number of epochs and batch size
history = model.fit(X_train, y_train, epochs=75, validation_data=(X_val, y_val), batch_size=64, verbose=1)

# Step 8: Evaluate the Model on Validation Data
val_loss, val_accuracy = model.evaluate(X_val, y_val)
y_pred = np.argmax(model.predict(X_val), axis=1)

# Step 9: Print Classification Report and Confusion Matrix
# Generate a classification report and confusion matrix for detailed evaluation
print("Classification Report:\n", classification_report(y_val, y_pred, target_names=[
    'Hand towards body', 'Hand down', 'Hand outwards', 'Hand upwards', 'Hand forward']))
print("Confusion Matrix:\n", confusion_matrix(y_val, y_pred))

# Step 10: Save the Model and Scaler
# Save the trained model and scaler for future use in prediction
model.save('complex_stroke_movement_classifier.h5')
joblib.dump(scaler, 'scaler.joblib')

# Step 11: Enhanced Plotting for Training and Validation Accuracy
# Visualize accuracy trends with professional styling
plt.figure(figsize=(10, 6))
plt.plot(history.history['accuracy'], label='Training Accuracy', linewidth=2)
plt.plot(history.history['val_accuracy'], label='Validation Accuracy', linestyle='--', linewidth=2)
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Accuracy', fontsize=12)
plt.title('Training and Validation Accuracy Over Epochs', fontsize=14)
plt.legend(loc='lower right', fontsize=10)
plt.grid(visible=True, linestyle='--', linewidth=0.5)
plt.tight_layout()
plt.show()

# Step 12: Enhanced Plotting for Training and Validation Loss
# Visualize loss trends with professional styling
plt.figure(figsize=(10, 6))
plt.plot(history.history['loss'], label='Training Loss', linewidth=2)
plt.plot(history.history['val_loss'], label='Validation Loss', linestyle='--', linewidth=2)
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Loss', fontsize=12)
plt.title('Training and Validation Loss Over Epochs', fontsize=14)
plt.legend(loc='upper right', fontsize=10)
plt.grid(visible=True, linestyle='--', linewidth=0.5)
plt.tight_layout()
plt.show()
