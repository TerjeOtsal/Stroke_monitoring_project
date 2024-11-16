import pandas as pd
import numpy as np
import tensorflow as tf
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt

# Step 1: Load and Preprocess Data for Training
data = pd.read_csv('combined_labeled_stroke_data.csv')  # Replace with the path to your labeled CSV file

# Separate features (X) and labels (y)
features = ['qw', 'qx', 'qy', 'qz', 'yaw', 'pitch', 'roll']  # Select only relevant features
X = data[features]
y = data['label'] - 1  # Adjust labels to start from 0 for TensorFlow compatibility

# Handle NaNs or infs
X = X.replace([np.inf, -np.inf], np.nan).fillna(0)

# Step 2: Standardize Features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split Data into Training and Validation Sets
X_train, X_val, y_train, y_val = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Save the scaler for use in testing script
joblib.dump(scaler, 'scaler.joblib')

# Step 3: Define the Feedforward Neural Network Model
def create_ffnn_model(input_shape):
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation='relu', input_shape=input_shape),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.3),
        
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.3),
        
        tf.keras.layers.Dense(16, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        
        tf.keras.layers.Dense(5, activation='softmax')  # Output layer for 5 classes
    ])
    
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005), 
                  loss='sparse_categorical_crossentropy', 
                  metrics=['accuracy'])
    return model

model = create_ffnn_model((X_train.shape[1],))

# Step 4: Train the Model with Early Stopping
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

history = model.fit(X_train, y_train, epochs=100, validation_data=(X_val, y_val), batch_size=32, verbose=1, 
                    callbacks=[early_stopping])

# Step 5: Evaluate the Model on Validation Data
val_loss, val_accuracy = model.evaluate(X_val, y_val)
y_pred = np.argmax(model.predict(X_val), axis=1)

# Print Classification Report and Confusion Matrix
print("Classification Report on Validation Data:\n", classification_report(y_val, y_pred, target_names=[
    'Hand towards body', 'Hand down', 'Hand outwards', 'Hand upwards', 'Hand forward']))
print("Confusion Matrix on Validation Data:\n", confusion_matrix(y_val, y_pred))

# Save the Model
model.save('ffnn_stroke_movement_classifier.keras')
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

# Step 6: Load and Test the Model on Activity.csv
def load_and_test_model(test_data_path):
    # Load the saved model and scaler
    loaded_model = tf.keras.models.load_model('ffnn_stroke_movement_classifier.keras')
    loaded_scaler = joblib.load('scaler.joblib')
    
    # Load and preprocess the test data
    test_data = pd.read_csv(test_data_path)
    X_test = test_data[features]

    # Check if 'label' column is present
    if 'label' in test_data.columns:
        y_test = test_data['label'] - 1  # Adjust labels to start from 0 for TensorFlow compatibility
        has_labels = True
    else:
        print("No 'label' column found in the test data. Running predictions only.")
        y_test = None
        has_labels = False
    
    # Scale the test data using the loaded scaler
    X_test_scaled = loaded_scaler.transform(X_test)

    # Make predictions
    y_test_pred = np.argmax(loaded_model.predict(X_test_scaled), axis=1)
    
    # Display predictions
    class_names = ['Hand towards body', 'Hand down', 'Hand outwards', 'Hand upwards', 'Hand forward']
    predicted_labels = [class_names[pred] for pred in y_test_pred]
    predictions_df = pd.DataFrame({
        'Predicted Class': y_test_pred,
        'Predicted Label': predicted_labels
    })
    print(predictions_df.head())

    # Plot distribution of predicted classes
    plt.figure(figsize=(10, 6))
    predictions_df['Predicted Label'].value_counts().plot(kind='bar', color='skyblue')
    plt.title('Distribution of Predicted Classes on Activity Data')
    plt.xlabel('Class')
    plt.ylabel('Frequency')
    plt.xticks(rotation=45)
    plt.show()

    # If labels are available, evaluate and print metrics
    if has_labels:
        test_loss, test_accuracy = loaded_model.evaluate(X_test_scaled, y_test)
        print(f"Test Loss: {test_loss}, Test Accuracy: {test_accuracy}")
        print("Test Classification Report:\n", classification_report(y_test, y_test_pred, target_names=class_names))
        print("Test Confusion Matrix:\n", confusion_matrix(y_test, y_test_pred))

# Test the model on Activity.csv
load_and_test_model('Activity.csv')  # Replace with your test CSV file path
