import pandas as pd
import numpy as np
import tensorflow as tf
import joblib
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt

def load_and_test_model(test_data_path):
    # Load the saved model and scaler
    loaded_model = tf.keras.models.load_model('ffnn_stroke_movement_classifier.keras')
    loaded_scaler = joblib.load('scaler.joblib')
    
    # Load and preprocess the test data
    test_data = pd.read_csv(test_data_path)
    features = ['qw', 'qx', 'qy', 'qz', 'yaw', 'pitch', 'roll']
    X_test = test_data[features]
    y_test = test_data['label'] - 1  # Adjust labels to start from 0 for TensorFlow compatibility
    
    # Scale the test data using the loaded scaler
    X_test_scaled = loaded_scaler.transform(X_test)

    # Make predictions (without allowing the model to see labels)
    y_test_pred = np.argmax(loaded_model.predict(X_test_scaled), axis=1)
    
    # Display predictions and plot distribution
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
    plt.title('Distribution of Predicted Classes on Test Data')
    plt.xlabel('Class')
    plt.ylabel('Frequency')
    plt.xticks(rotation=45)
    plt.show()

    # Evaluate model's performance against true labels
    test_loss, test_accuracy = loaded_model.evaluate(X_test_scaled, y_test)
    print(f"Test Loss: {test_loss}, Test Accuracy: {test_accuracy}")
    print("Test Classification Report:\n", classification_report(y_test, y_test_pred, target_names=class_names))
    print("Test Confusion Matrix:\n", confusion_matrix(y_test, y_test_pred))

# Test the model on the new labeled test dataset
load_and_test_model('combined_labeled_stroke_data3.csv')  # Replace with the actual file path if different
