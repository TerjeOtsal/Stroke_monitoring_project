import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import joblib

# Load the labeled data
data = pd.read_csv('refined_synthetic_stroke_data_rounded.csv')

# Separate features and labels
X = data.drop(columns=['label'])
y = data['label']

# Normalize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split (80-20 split)
X_train, X_val, y_train, y_val = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Define and train the Random Forest classifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Model evaluation
y_pred = model.predict(X_val)
print("Classification Report:\n", classification_report(y_val, y_pred, target_names=[
    'Hand towards body', 'Hand down', 'Hand outwards', 'Hand upwards', 'Hand forward']))
print("Confusion Matrix:\n", confusion_matrix(y_val, y_pred))

# Save the trained model and the scaler
joblib.dump(model, 'stroke_movement_classifier.joblib')
joblib.dump(scaler, 'scaler.joblib')

print("Model and scaler saved as 'stroke_movement_classifier.joblib' and 'scaler.joblib'")
