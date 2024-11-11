import pandas as pd

# Load each class file
class1_df = pd.read_csv('Class1.csv')
class2_df = pd.read_csv('Class2.csv')
class3_df = pd.read_csv('Class3.csv')
class4_df = pd.read_csv('Class4.csv')
class5_df = pd.read_csv('Class5.csv')

# Label each class with appropriate class number
class1_df['label'] = 1
class2_df['label'] = 2
class3_df['label'] = 3
class4_df['label'] = 4
class5_df['label'] = 5

# Concatenate all class data into a single DataFrame
labeled_data = pd.concat([class1_df, class2_df, class3_df, class4_df, class5_df], ignore_index=True)

# Save the labeled data as a single CSV file for training and validation purposes
labeled_data.to_csv('combined_labeled_stroke_data.csv', index=False)

print("Combined labeled data saved as 'combined_labeled_stroke_data.csv'")
