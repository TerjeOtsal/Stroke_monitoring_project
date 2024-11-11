import pandas as pd
import numpy as np
import tensorflow as tf
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt

# Step 1: Load and Preprocess Data
# Load labeled CSV data with the expected structure, including 'label' column for classes
data = pd.read_csv('combined_labeled_stroke_data.csv')  # Replace with the path to your labeled CSV file

# Separate features (X) and labels (y)
X = data.drop(columns=['label'])  # Features only
y = data['label'] - 1  # Adjust labels to start from 0 for TensorFlow compatibility

# Check for NaN or infinite values in data
if np.any(np.isnan(X)) or np.any(np.isinf(X)):
    print("Found NaN or infinite values in data; replacing with zeros.")
    X = X.fillna(0)  # Replace NaNs or infs with zero or another strategy

# Step 2: Standardize Features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 3: Reshape Data for CNN
# Define the number of timesteps, which is the sequence length each input sample will use
timesteps = 10  # This can be adjusted based on your data characteristics
X_reshaped = np.array([X_scaled[i:i + timesteps] for i in range(len(X_scaled) - timesteps)])
y_reshaped = y[timesteps:].values  # Align y to match reshaped X

# Step 4: Split Data into Training and Validation Sets
X_train, X_val, y_train, y_val = train_test_split(X_reshaped, y_reshaped, test_size=0.2, random_state=42)

# Step 5: Define a Simple CNN Model Architecture
model = tf.keras.Sequential([
    # First Conv1D layer with Batch Normalization and MaxPooling
    tf.keras.layers.Conv1D(filters=32, kernel_size=3, activation='relu', input_shape=(timesteps, X_reshaped.shape[2])),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling1D(pool_size=2),
    
    # Second Conv1D layer with Batch Normalization and MaxPooling
    tf.keras.layers.Conv1D(filters=16, kernel_size=3, activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling1D(pool_size=2),
    
    # Flatten layer to convert the 2D features to 1D for dense layers
    tf.keras.layers.Flatten(),
    
    # Dense layer with Dropout for regularization
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dropout(0.3),
    
    # Output layer with softmax activation for multi-class classification (5 classes)
    tf.keras.layers.Dense(5, activation='softmax')
])

# Step 6: Compile the Model with a Lower Learning Rate and Gradient Clipping
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001, clipvalue=1.0), 
              loss='sparse_categorical_crossentropy', 
              metrics=['accuracy'])

# Step 7: Train the Model
history = model.fit(X_train, y_train, epochs=150, validation_data=(X_val, y_val), batch_size=64, verbose=1)

# Step 8: Evaluate the Model on Validation Data
val_loss, val_accuracy = model.evaluate(X_val, y_val)
y_pred = np.argmax(model.predict(X_val), axis=1)

# Step 9: Print Classification Report and Confusion Matrix
print("Classification Report:\n", classification_report(y_val, y_pred, target_names=[
    'Hand towards body', 'Hand down', 'Hand outwards', 'Hand upwards', 'Hand forward']))
print("Confusion Matrix:\n", confusion_matrix(y_val, y_pred))

# Step 10: Save the Model and Scaler
model.save('cnn_stroke_movement_classifier.keras')
joblib.dump(scaler, 'scaler.joblib')
print("Model and scaler saved.")

# Plot Training and Validation Accuracy
plt.figure(figsize=(10, 6))
plt.plot(history.history['accuracy'], label='Training Accuracy', linewidth=2)
plt.plot(history.history['val_accuracy'], label='Validation Accuracy', linestyle='--', linewidth=2)
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Training and Validation Accuracy')
plt.legend(loc='lower right')
plt.grid(True)
plt.tight_layout()
plt.show()

# Plot Training and Validation Loss
plt.figure(figsize=(10, 6))
plt.plot(history.history['loss'], label='Training Loss', linewidth=2)
plt.plot(history.history['val_loss'], label='Validation Loss', linestyle='--', linewidth=2)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend(loc='upper right')
plt.grid(True)
plt.tight_layout()
plt.show()

# Additional Functionality: Load and Test the Model Later

def load_and_test_model(test_data_path):
    # Load the saved model and scaler
    loaded_model = tf.keras.models.load_model('cnn_stroke_movement_classifier.keras')
    loaded_scaler = joblib.load('scaler.joblib')
    
    # Load and preprocess the test data
    test_data = pd.read_csv(test_data_path)
    X_test = test_data.drop(columns=['label'])
    y_test = test_data['label'] - 1
    
    # Scale the test data using the loaded scaler
    X_test_scaled = loaded_scaler.transform(X_test)
    
    # Reshape the data to match the model's expected input
    X_test_reshaped = np.array([X_test_scaled[i:i + timesteps] for i in range(len(X_test_scaled) - timesteps)])
    y_test_reshaped = y_test[timesteps:].values

    # Evaluate the model on the test data
    test_loss, test_accuracy = loaded_model.evaluate(X_test_reshaped, y_test_reshaped)
    print(f"Test Loss: {test_loss}, Test Accuracy: {test_accuracy}")

    # Make predictions and print classification report
    y_test_pred = np.argmax(loaded_model.predict(X_test_reshaped), axis=1)
    print("Test Classification Report:\n", classification_report(y_test_reshaped, y_test_pred))
    print("Test Confusion Matrix:\n", confusion_matrix(y_test_reshaped, y_test_pred))

# Example usage:
# load_and_test_model('new_test_data.csv')  # Replace with your test CSV file path
