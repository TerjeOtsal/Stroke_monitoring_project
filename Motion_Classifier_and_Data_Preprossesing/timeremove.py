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

    # Save the cleaned data to a new CSV file
    data.to_csv(output_file, index=False)
    print(f"Cleaned data saved to {output_file}.")

# Usage
clean_csv('BatteryTest.csv', 'BatteryTest2.csv')
