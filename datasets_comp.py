import pandas as pd

def compare_csv(file1, file2):
    # Read CSV files into pandas DataFrames
    df1 = pd.read_csv(file1)
    df2 = pd.read_csv(file2)

    # Check if the column names are the same in both DataFrames
    if set(df1.columns) != set(df2.columns):
        print("Column names are different. Cannot compare.")
        return

    # Check for differences in the data
    differences = df1.compare(df2)

    if differences.empty:
        print("No differences found.")
    else:
        print("Differences found:")
        print(differences)

# Example usage
file1_path = 'new_dataset.csv'  # Replace with the actual path to your first CSV file
file2_path = 'preprocessed_new_dataset.csv'  # Replace with the actual path to your second CSV file

compare_csv(file1_path, file2_path)
