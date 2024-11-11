import pandas as pd

def clean_csv(input_file, output_file, label_column='label'):
    """
    Cleans the CSV file by removing the 'Timestamp' column if it exists 
    and dropping any rows without a corresponding label.

    Parameters:
    - input_file: Path to the original CSV file.
    - output_file: Path to save the cleaned CSV file.
    - label_column: The name of the column that contains labels. Rows missing this column's value are removed.
    """
    # Load the CSV file
    data = pd.read_csv(input_file)
    
    # Drop the 'Timestamp' column if it exists
    if 'Timestamp' in data.columns:
        data = data.drop(columns=['Timestamp'])
        print("Removed 'Timestamp' column.")

    # Drop rows where the label column has missing values
    initial_row_count = len(data)
    data = data.dropna(subset=[label_column])
    removed_rows = initial_row_count - len(data)
    print(f"Removed {removed_rows} rows with missing labels in '{label_column}' column.")

    # Save the cleaned data to a new CSV file
    data.to_csv(output_file, index=False)
    print(f"Cleaned data saved to {output_file}.")

# Usage
clean_csv('activity2.csv', 'activity2.csv')
