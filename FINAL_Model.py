import pandas as pd
import numpy as np
import tensorflow as tf
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns

# Define the class names once for global use
class_names = ['Hand towards body', 'Hand down', 'Hand outwards', 'Hand upwards', 'Hand forward']

# Step 1: Load and Combine Datasets
data1 = pd.read_csv('used csvfiles/combined_labeled_stroke_data.csv')
data2 = pd.read_csv('used csvfiles/combined_labeled_stroke_data3.csv')

# Concatenate datasets and shuffle
combined_data = pd.concat([data1, data2], ignore_index=True)

# Function to balance classes by sampling equal numbers from each class
def balance_classes(data, label_column):
    groups = data.groupby(label_column)
    min_size = groups.size().min()
    balanced_data = groups.apply(lambda x: x.sample(min_size, random_state=42)).reset_index(drop=True)
    return balanced_data

# Balance the dataset
balanced_data = balance_classes(combined_data, label_column='label')

# Separate features and labels
features = ['qw', 'qx', 'qy', 'qz', 'yaw', 'pitch', 'roll']
X = balanced_data[features]
y = balanced_data['label'] - 1  # Adjust labels to start from 0

# Step 2: Train, Validation, and Test Split
X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.25, random_state=42, stratify=y_temp)

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
        # First hidden layer with more neurons
        tf.keras.layers.Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001), input_shape=input_shape),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.5),
        
        # Second hidden layer
        tf.keras.layers.Dense(64, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.4),
        
        # Third hidden layer
        tf.keras.layers.Dense(64, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.3),
        
        # Fourth hidden layer
        tf.keras.layers.Dense(32, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.2),
        
        # Output layer
        tf.keras.layers.Dense(5, activation='softmax')  # Output layer with 5 classes
    ])
    
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4, clipvalue=1.0), 
                  loss='sparse_categorical_crossentropy', 
                  metrics=['accuracy'])
    return model

model = create_ffnn_model((X_train_scaled.shape[1],))

# Learning rate scheduler
lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=15, min_lr=1e-6)

# Early stopping callback
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=25, restore_best_weights=True)

# Step 6: Train the Model with Class Weights
history = model.fit(X_train_scaled, y_train_array, epochs=2000, validation_data=(X_val_scaled, y_val), 
                    batch_size=64, verbose=1, class_weight=class_weight_dict,
                    callbacks=[early_stopping, lr_scheduler])

# Save the trained model
model.save('ffnn_stroke_movement_classifier_v4.keras')
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

# Function to generate a performance matrix
def generate_performance_matrix(y_true, y_pred, class_names):
    """
    Generates a DataFrame displaying precision, recall, F1 score, and support for each class.
    """
    report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
    performance_df = pd.DataFrame(report).transpose()
    performance_df = performance_df[['precision', 'recall', 'f1-score', 'support']]
    return performance_df


# Step 8: Evaluate the Model on Test Data
def evaluate_model_with_matrix(X_test_scaled, y_test):
    loaded_model = tf.keras.models.load_model('ffnn_stroke_movement_classifier_v4.keras')
    y_test_pred = np.argmax(loaded_model.predict(X_test_scaled), axis=1)

    # Performance matrix
    performance_matrix = generate_performance_matrix(y_test, y_test_pred, class_names)
    print("\nPerformance Matrix:\n")
    print(performance_matrix)

    # Plot the performance matrix
    plt.figure(figsize=(12, 8))
    sns.heatmap(performance_matrix.iloc[:-1, :-1], annot=True, fmt=".2f", cmap="coolwarm", linewidths=0.5, cbar_kws={'label': 'Score'})
    plt.title('Performance Matrix Heatmap', fontsize=16)
    plt.xlabel('Metrics', fontsize=12)
    plt.ylabel('Classes', fontsize=12)
    plt.tight_layout()
    plt.show()

    # Confusion Matrix
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

# Test the model and generate the performance matrix
evaluate_model_with_matrix(X_test_scaled, y_test)
