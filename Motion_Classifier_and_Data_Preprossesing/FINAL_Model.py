import pandas as pd
import numpy as np
import tensorflow as tf
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Define the class names once for global use
class_names = ['Hand towards body', 'Hand down', 'Hand outwards', 'Hand upwards', 'Hand forward']

# Step 1: Load and Combine Datasets
data1 = pd.read_csv('used csvfiles\combined_labeled_stroke_data.csv')
data2 = pd.read_csv('used csvfiles\combined_labeled_stroke_data3.csv')

# Concatenate datasets and shuffle
combined_data = pd.concat([data1, data2], ignore_index=True).sample(frac=1, random_state=42)

# Separate features and labels
features = ['qw', 'qx', 'qy', 'qz', 'yaw', 'pitch', 'roll']
X = combined_data[features]
y = combined_data['label'] - 1  # Adjust labels to start from 0

# Step 2: Train, Validation, and Test Split
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Remove NaN and infinite values
X_train = X_train.replace([np.inf, -np.inf], np.nan).fillna(0)
X_val = X_val.replace([np.inf, -np.inf], np.nan).fillna(0)
X_test = X_test.replace([np.inf, -np.inf], np.nan).fillna(0)

# Step 3: Standardize Features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)
joblib.dump(scaler, 'scaler.joblib')

# Step 4: Calculate Class Weights
y_train_array = y_train.to_numpy()
class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y_train_array), y=y_train_array)
class_weight_dict = {i: class_weights[i] for i in range(len(class_weights))}

# Step 5: Define a Model with Gradient Clipping and Lower Learning Rate
def create_ffnn_model(input_shape):
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001), input_shape=input_shape),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.4),
        
        tf.keras.layers.Dense(32, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.3),
        
        tf.keras.layers.Dense(5, activation='softmax')
    ])
    
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4, clipvalue=1.0), 
                  loss='sparse_categorical_crossentropy', 
                  metrics=['accuracy'])
    return model

model = create_ffnn_model((X_train_scaled.shape[1],))

# Learning rate scheduler
lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)

# Early stopping callback
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)

# Step 6: Train the Model with Class Weights
history = model.fit(X_train_scaled, y_train_array, epochs=2000, validation_data=(X_val_scaled, y_val), 
                    batch_size=64, verbose=1, class_weight=class_weight_dict,
                    callbacks=[early_stopping, lr_scheduler])

# Save the trained model
model.save('ffnn_stroke_movement_classifier_v3.keras')
print("Model and scaler saved.")

# Step 7: Plot Training and Validation Accuracy and Loss
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

# Step 8: Evaluate the Model on Test Data
def evaluate_model(X_test_scaled, y_test):
    loaded_model = tf.keras.models.load_model('ffnn_stroke_movement_classifier_v3.keras')
    
    y_test_pred = np.argmax(loaded_model.predict(X_test_scaled), axis=1)
    
    print("Classification Report on Test Data:\n", classification_report(y_test, y_test_pred, target_names=class_names))
    print("Confusion Matrix on Test Data:\n", confusion_matrix(y_test, y_test_pred))

    predicted_labels = [class_names[pred] for pred in y_test_pred]
    predictions_df = pd.DataFrame({'Predicted Class': y_test_pred, 'Predicted Label': predicted_labels})
    
    plt.figure(figsize=(10, 6))
    predictions_df['Predicted Label'].value_counts().plot(kind='bar', color='skyblue')
    plt.title('Distribution of Predicted Classes on Test Data')
    plt.xlabel('Class')
    plt.ylabel('Frequency')
    plt.xticks(rotation=45)
    plt.show()

    # Plot the confusion matrix
    cm = confusion_matrix(y_test, y_test_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='g', cmap='coolwarm', xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Number of Predictions'}, linewidths=0.5, linecolor='black')
    plt.title('Confusion Matrix', fontsize=16)
    plt.xlabel('Predicted Class', fontsize=12)
    plt.ylabel('True Class', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()

evaluate_model(X_test_scaled, y_test)

# Step 9: Test on a New CSV File
def test_on_new_csv(test_data_path):
    # Load model and scaler
    loaded_model = tf.keras.models.load_model('ffnn_stroke_movement_classifier_v3.keras')
    loaded_scaler = joblib.load('scaler.joblib')
    
    # Load test data and preprocess
    test_data = pd.read_csv(test_data_path)
    if 'label' in test_data.columns:
        y_true = test_data['label'] - 1  # True labels (if available)
    else:
        y_true = None
    
    # Ensure feature columns exist and process the data
    X_test_new = test_data[features].replace([np.inf, -np.inf], np.nan).fillna(0)
    X_test_new_scaled = loaded_scaler.transform(X_test_new)
    
    # Predict
    y_pred_new = np.argmax(loaded_model.predict(X_test_new_scaled), axis=1)
    predicted_labels = [class_names[pred] for pred in y_pred_new]
    
    # Display predictions sequentially
    plt.figure(figsize=(12, 6))
    plt.plot(range(len(y_pred_new)), y_pred_new, marker='o', color='skyblue')
    plt.title('Sequential Predictions on New Test Data')
    plt.xlabel('Index')
    plt.ylabel('Predicted Class')
    plt.yticks(ticks=range(len(class_names)), labels=class_names)
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    
    if y_true is not None:
        # Evaluate if true labels are available
        print("Classification Report:\n", classification_report(y_true, y_pred_new, target_names=class_names))
        cm = confusion_matrix(y_true, y_pred_new)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='g', cmap='coolwarm', xticklabels=class_names, yticklabels=class_names)
        plt.title('Confusion Matrix for New Data')
        plt.xlabel('Predicted Class')
        plt.ylabel('True Class')
        plt.tight_layout()
        plt.show()

# Test the model on a new CSV file
test_on_new_csv('used csvfiles\BatteryTest2.csv')  # Replace with the actual file path if different
