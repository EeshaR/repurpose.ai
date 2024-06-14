import pandas as pd

# Load the data
file_path = '/repurpose.ai/data.txt'
data = pd.read_csv(file_path, sep='\t')

# Data Cleaning
def clean_data(data):
    # Handle missing values by filling with a placeholder (e.g., 'Unknown')
    data = data.fillna('Unknown')

    # Remove duplicates
    data = data.drop_duplicates()

    return data

# Data Organization
def organize_data(cleaned_data):
    # Select all columns for prediction
    columns = ['Name', 'MOA', 'Target', 'Disease Area', 'Indication', 'Vendor', 'Purity', 'Id', 'SMILES', 'InChIKey', 'Phase', 'Deprecated ID']
    organized_data = cleaned_data[columns]

    return organized_data

# Save the processed data to a file
def save_processed_data(organized_data, output_path):
    organized_data.to_csv(output_path, index=False)

if __name__ == '__main__':
    cleaned_data = clean_data(data)
    organized_data = organize_data(cleaned_data)
    save_processed_data(organized_data, 'processed_data.csv')
