import pandas as pd

# Load the CSV file
data = pd.read_csv('combined_labeled_stroke_data.csv')

# Check if 'timestamp' or any similar column name exists and drop it
# Replace 'timestamp' with the exact column name if it’s different
if 'heading' in data.columns:
    data = data.drop(columns=['heading'])
elif 'Heading' in data.columns:  # Check for other possible capitalizations
    data = data.drop(columns=['Heading'])

# Save the cleaned data to a new CSV file
data.to_csv('combined_labeled_stroke_data.csv', index=False)

print("Timestamp column removed, and data saved as 'your_file_no_timestamp.csv'")
