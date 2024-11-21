import pandas as pd
import numpy as np
import tensorflow as tf
import joblib
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns  # Import seaborn for the confusion matrix plot

def load_and_test_model(test_data_path):
    # Load the saved model and scaler
    loaded_model = tf.keras.models.load_model('ffnn_stroke_movement_classifier_v3.keras')
    loaded_scaler = joblib.load('scaler.joblib')
    
    # Load and preprocess the test data
    test_data = pd.read_csv(test_data_path)
    features = ['qw', 'qx', 'qy', 'qz', 'yaw', 'pitch', 'roll']
    
    # Check if required columns exist
    missing_columns = [col for col in features if col not in test_data.columns]
    if missing_columns:
        print(f"Error: Missing columns in the input file: {missing_columns}")
        return
    
    # Select features and convert to numeric
    X_test = test_data[features].apply(pd.to_numeric, errors='coerce').fillna(0)
    
    # Scale the test data using the loaded scaler
    X_test_scaled = loaded_scaler.transform(X_test)

    # Make predictions
    y_test_pred = np.argmax(loaded_model.predict(X_test_scaled), axis=1)
    
    # Map predictions to class names
    class_names = ['Hand towards body', 'Hand down', 'Hand outwards', 'Hand upwards', 'Hand forward']
    predicted_labels = [class_names[pred] for pred in y_test_pred]
    
    # Add predictions to the test data for display
    test_data['Predicted Label'] = predicted_labels

    # Display sample predictions as class names
    print("Sample Predictions (Class Names):")
    print(test_data[['Predicted Label']].head())

    # Plot sequential predictions
    plt.figure(figsize=(10, 6))
    plt.plot(range(len(test_data)), [class_names[pred] for pred in y_test_pred], marker='o', linestyle='-', color='skyblue', label="Predicted Class")
    plt.title('Predicted Movement Class Over Test Data (Sequential)')
    plt.xlabel('Index')
    plt.ylabel('Predicted Class')
    plt.xticks(rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Check if 'label' column exists for actual evaluation
    if 'label' in test_data.columns:
        y_test = test_data['label'] - 1  # Adjust labels to match model expectations
        
        # Classification report and confusion matrix
        print("Test Classification Report:\n", classification_report(y_test, y_test_pred, target_names=class_names))
        print("Test Confusion Matrix:\n", confusion_matrix(y_test, y_test_pred))
        
        # Confusion matrix plot
        cm = confusion_matrix(y_test, y_test_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='g', cmap='coolwarm', xticklabels=class_names, yticklabels=class_names,
                    cbar_kws={'label': 'Number of Predictions'}, linewidths=0.5, linecolor='black')
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted Class')
        plt.ylabel('True Class')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.show()
        
    else:
        print("No true labels available for evaluation; only predictions are displayed.")

# Run the model on the test dataset
load_and_test_model('used csvfiles/BatteryTest2.csv')  # Replace with the actual path to your test file

